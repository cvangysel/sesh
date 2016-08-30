from __future__ import unicode_literals

from cvangysel import io_utils, rank_utils
from sesh import domain, sesh_pb2

import collections
import json
import logging
import nltk
import numpy as np
import operator
import string


def create_scorer(scorer_desc, qrels_per_session=None):
    return SESSION_SCORERS[scorer_desc.type](
        scorer_desc, qrels_per_session)


class SessionScorer(object):

    def __init__(self, desc, qrels_per_session, encoding='utf8'):
        self.desc = desc
        self.DO_NOT_USE_qrels = qrels_per_session

        self.encoding = encoding

    def __repr__(self):
        return '<SessionScorer {0}>'.format(str(self.desc).replace('\n', ' '))

    def score_documents_for_session(
            self,
            session,
            candidate_document_ids,
            index, dictionary,
            anchor_texts,
            word_frequency_index,
            logger,
            f_debug_out):
        raise NotImplementedError()


class QrelOracleScorer(SessionScorer):

    def score_documents_for_session(
            self,
            session,
            candidate_document_ids,
            index, dictionary,
            anchor_texts,
            word_frequency_index,
            logger,
            f_debug_out):
        for document_id, relevance in \
                self.DO_NOT_USE_qrels[0][session.session_id]:
            yield document_id, relevance


class TermWeightingOracleScorer(SessionScorer):

    def __init__(self, *args, **kwargs):
        super(TermWeightingOracleScorer, self).__init__(*args, **kwargs)

        assert self.desc.term_reweighting_oracle_scorer_desc

        assert self.desc.term_reweighting_oracle_scorer_desc.\
            approximation_method in (
                sesh_pb2.TermWeightingOracleScorerDesc.RANKSVM,
                sesh_pb2.TermWeightingOracleScorerDesc.BRUTE)

    def score_documents_for_session(
            self,
            session,
            candidate_document_ids,
            index, dictionary,
            anchor_texts,
            word_frequency_index,
            logger,
            f_debug_out):
        document_ids = index.document_ids(candidate_document_ids)
        session_relevance = dict(self.DO_NOT_USE_qrels[0][session.session_id])

        terms = set([
            term
            for query in session.queries
            for term in query])

        document_features = np.zeros(
            (len(document_ids), len(terms)))

        for term_idx, term in enumerate(terms):
            results = dict(index.query(
                term,
                document_set=map(operator.itemgetter(1), document_ids)))

            for document_idx, (ext_document_id, int_document_id) in \
                    enumerate(document_ids):
                document_features[document_idx, term_idx] = \
                    results[int_document_id]

        document_relevances = np.zeros(len(document_ids))

        for document_idx, (document_id, _) in enumerate(document_ids):
            document_relevances[document_idx] = session_relevance.get(
                document_id, 0.0)

        # If we have no relevant documents, then skip.
        if np.all(document_relevances <= 0.0):
            return

        if self.desc.term_reweighting_oracle_scorer_desc\
            .approximation_method == \
                sesh_pb2.TermWeightingOracleScorerDesc.RANKSVM:
            import pysofia
            import pysofia.compat

            svm = pysofia.compat.RankSVM(max_iter=10000)

            # Fit the SVM.
            svm.fit(document_features, document_relevances)

            term_weights = svm.coef_
        elif self.desc.term_reweighting_oracle_scorer_desc\
            .approximation_method == \
                sesh_pb2.TermWeightingOracleScorerDesc.BRUTE:
            term_weights = rank_utils.optimal_weight_vector(
                document_features, document_relevances, rank_cutoff=10)

            if term_weights is None:
                logging.error('Unable to compute optimal weights.')

                return

        f_debug_out.write('Terms: {}\n'.format(terms))
        f_debug_out.write('Weights: {}\n'.format(term_weights))

        results = document_features.dot(term_weights)

        for document_idx, (document_id, _) in enumerate(document_ids):
            yield document_id, results[document_idx]


class IndriScorer(SessionScorer):

    def __init__(self, *args, **kwargs):
        super(IndriScorer, self).__init__(*args, **kwargs)

        self.exclude = set(string.punctuation)

        assert self.desc.indri_scorer_desc
        assert self.desc.indri_scorer_desc.query_position in (
            sesh_pb2.IndriScorerDesc.FIRST,
            sesh_pb2.IndriScorerDesc.LAST,
            sesh_pb2.IndriScorerDesc.ALL,
            sesh_pb2.IndriScorerDesc.THEME,
            sesh_pb2.IndriScorerDesc.CUSTOM)

        if self.desc.indri_scorer_desc.query_position == \
                sesh_pb2.IndriScorerDesc.CUSTOM:
            assert self.desc.indri_scorer_desc.weights_path

            with open(self.desc.indri_scorer_desc.weights_path, 'r') \
                    as f_weights:
                self.term_weights = json.loads(f_weights.read())
        else:
            self.term_weights = None

    def score_documents_for_session(
            self,
            session,
            candidate_document_ids,
            index, dictionary,
            anchor_texts,
            word_frequency_index,
            logger,
            f_debug_out):
        if self.desc.indri_scorer_desc.query_position in (
                sesh_pb2.IndriScorerDesc.FIRST,
                sesh_pb2.IndriScorerDesc.LAST):
            query_idx = (
                -1
                if self.desc.indri_scorer_desc.query_position ==
                sesh_pb2.IndriScorerDesc.LAST
                else 0)

            if self.desc.indri_scorer_desc.cleaned:
                terms = session.interactions[query_idx].query
            else:
                terms = session.interactions[query_idx].original_query.split()

            if self.desc.indri_scorer_desc.binary:
                terms = list(set(terms))

            query = session.interactions[query_idx].original_query
        elif self.desc.indri_scorer_desc.query_position == \
                sesh_pb2.IndriScorerDesc.ALL:
            if self.desc.indri_scorer_desc.cleaned:
                terms = [
                    term
                    for interaction in session.interactions
                    for term in interaction.query]
            else:
                terms = [
                    term
                    for interaction in session.interactions
                    for term in interaction.original_query.split()]

            if self.desc.indri_scorer_desc.binary:
                terms = list(set(terms))

            query = ' '.join(terms)
        elif self.desc.indri_scorer_desc.query_position == \
                sesh_pb2.IndriScorerDesc.THEME:
            theme_terms = set(session.queries[0])

            for query in session.queries[1:]:
                theme_terms &= set(query)

            if theme_terms:
                query = ' '.join(theme_terms)
            else:
                query = ' '.join(
                    interaction.original_query
                    for interaction in session.interactions)
        elif self.desc.indri_scorer_desc.query_position == \
                sesh_pb2.IndriScorerDesc.CUSTOM:
            assert self.term_weights is not None

            if (str(session.track_edition) not in self.term_weights or
                    str(session.session_id) not in
                    self.term_weights[str(session.track_edition)]):
                return

            query_pieces = []

            for term, weight in self.term_weights[
                    str(session.track_edition)][
                    str(session.session_id)].items():
                query_pieces.append('{:.20f} {}'.format(weight, term))

            query = '#weight( {} )'.format(' '.join(query_pieces))
        else:
            raise NotImplementedError()

        # Strip punctuation.
        if self.desc.indri_scorer_desc.query_position != \
                sesh_pb2.IndriScorerDesc.CUSTOM:
            query = ''.join(ch for ch in query if ch not in self.exclude)

        f_debug_out.write('Query: {}\n'.format(query))

        logger.debug('Indri query: %s', query)
        results = index.query(
            query,
            document_set=map(operator.itemgetter(1),
                             index.document_ids(candidate_document_ids)))

        logger.debug('Indri returned %d results.', len(results))

        for int_document_id, log_score in results:
            ext_document_id, _ = index.document(int_document_id)

            yield ext_document_id, log_score


class NuggetScorer(SessionScorer):

    def __init__(self, *args, **kwargs):
        super(NuggetScorer, self).__init__(*args, **kwargs)

        assert self.desc.nugget_scorer_desc

        assert self.desc.nugget_scorer_desc.reading_level in (
            # sesh_pb2.NuggetScorerDesc.RL1,
            sesh_pb2.NuggetScorerDesc.RL2,
            sesh_pb2.NuggetScorerDesc.RL3,
            sesh_pb2.NuggetScorerDesc.RL4)

        self.top_k = 10
        self.normalized_occurrence_threshold = 0.97

        self.top_anchor_texts = 5
        self.anchor_texts_beta = 0.1

    def score_documents_for_session(
            self,
            session,
            candidate_document_ids,
            index, dictionary,
            anchor_texts,
            word_frequency_index,
            logger,
            f_debug_out):
        def tokenize_fn(text):
            return [
                token
                for token in io_utils.tokenize_text(text)
                if dictionary.has_token(token)]

        def translate_fn(text):
            return [
                dictionary.translate_token(token)
                for token in tokenize_fn(text)]

        #
        # Query expansion with previous search results (RL3/RL4).
        #
        # Add anchor texts occurring in the document collection
        # to pages that appeared in some SERP of the session.
        #
        # For RL4, only the anchor texts of pages that were clicked
        # are considered.
        #

        top_anchor_normalized_frequencies_and_texts = []

        if (self.desc.nugget_scorer_desc.reading_level <
                sesh_pb2.NuggetScorerDesc.RL3):
            pass  # Ignore anchor texts.
        elif not anchor_texts:
            logger.error('No anchor texts available. Nugget method will only '
                         'use information of pervious queries (RL2).')
        else:
            doc_ids_and_urls = {
                document.doc_id: document.url
                for interaction in filter(
                    lambda interaction: interaction.serp is not None,
                    session.interactions)
                for document in interaction.serp
                if document.url in anchor_texts}

            if (self.desc.nugget_scorer_desc.reading_level ==
                    sesh_pb2.NuggetScorerDesc.RL4):
                clicked_doc_ids = set(
                    doc_id
                    for interaction in filter(
                        lambda interaction: interaction.clicks is not None,
                        session.interactions)
                    for doc_id in interaction.clicks)

                logger.debug('Clicked DocIDs: %s', clicked_doc_ids)

                urls = [url for doc_id, url in doc_ids_and_urls.items()
                        if doc_id in clicked_doc_ids]
            else:
                urls = doc_ids_and_urls.values()

            logger.debug('Including anchor texts of %s in query.', urls)

            def _generate_anchor_text_queries():
                for anchor_text in (anchor_text for url in urls
                                    for anchor_text in anchor_texts[url]):
                    anchor_text_tokens = tokenize_fn(anchor_text)

                    if not anchor_text_tokens:
                        continue

                    yield '#combine({0})'.format(' '.join(
                        token for token in anchor_text_tokens))

            anchor_text_queries_and_frequencies = collections.Counter(
                _generate_anchor_text_queries())

            if anchor_text_queries_and_frequencies:
                partition_fn = float(
                    max(anchor_text_queries_and_frequencies.values()))

                anchor_texts_and_frequencies = sorted(
                    anchor_text_queries_and_frequencies.items(),
                    key=operator.itemgetter(1),
                    reverse=True)

                anchor_normalized_frequencies_and_texts = [
                    ((frequency / partition_fn), anchor_text)
                    for anchor_text, frequency in
                    anchor_texts_and_frequencies]

                top_anchor_normalized_frequencies_and_texts = \
                    anchor_normalized_frequencies_and_texts[
                        :self.top_anchor_texts]

        #
        # Query expansion with previous queries (RL2).
        #
        # Score every query in the session using the Indri index
        # and also retrieve the snippets of the new SERP.
        #
        # Add bigrams and higher-order N-grams from all queries
        # based on their occurrence in their Indri SERP.
        #
        # Currently only implements the strict expansion method.
        #

        expanded_queries = []

        if (self.desc.nugget_scorer_desc.reading_level <
                sesh_pb2.NuggetScorerDesc.RL2):
            pass  # Ignore previous queries.
        else:
            for query in session.queries:
                if not query:
                    logger.debug('Skipping query as it is empty.')

                    continue

                indri_query = ' '.join(query)

                # TODO(cvangysel): should we also restrict the
                # following query() call to the candidate document set?
                results = index.query(
                    indri_query,
                    results_requested=self.top_k, include_snippets=True)

                reference_document_tokens = []

                for int_document_id, score, snippet in results:
                    snippet_tokens = translate_fn(snippet)
                    reference_document_tokens.extend(snippet_tokens)

                unigram_frequencies = collections.Counter(
                    reference_document_tokens)

                bigram_frequencies = collections.Counter(
                    nltk.bigrams(reference_document_tokens))

                query_tokens = translate_fn(indri_query)

                nuggets = set()

                for bigram in nltk.bigrams(query_tokens):
                    frequency = bigram_frequencies.get(bigram, 0)

                    if not frequency:
                        continue

                    w_context, w_current = bigram

                    max_frequency = min(
                        unigram_frequencies[w_context],
                        unigram_frequencies[w_current])

                    normalized_occurrence = \
                        float(frequency) / float(max_frequency)

                    if normalized_occurrence >= \
                            self.normalized_occurrence_threshold:
                        nuggets.add(bigram)

                while True:
                    new_nuggets = set()

                    for first_nugget in nuggets:
                        for second_nugget in nuggets:
                            if not first_nugget or not second_nugget:
                                continue

                            if first_nugget == second_nugget:
                                continue

                            if first_nugget[-1] == second_nugget[0] and \
                                    first_nugget[0] != second_nugget[-1]:
                                new_nugget = first_nugget + second_nugget[1:]

                                if new_nugget in nuggets:
                                    continue

                                logger.debug('Discovered expanded nugget %s '
                                             'by concatenating %s and %s.',
                                             new_nugget,
                                             first_nugget, second_nugget)

                                new_nuggets.add(new_nugget)

                    if new_nuggets:
                        nuggets.update(new_nuggets)
                    else:
                        break

                logger.debug('Found %d nuggets for query %s in %s.',
                             len(nuggets), query, session)

                expanded_queries.append(
                    '#combine({nuggets} {words})'.format(
                        nuggets=' '.join(
                            '#1({0})'.format(
                                ' '.join(dictionary[token_id]
                                         for token_id in nugget))
                            for nugget in nuggets),
                        words=' '.join(query)))

        #
        # Combine everything in a big Indri query.
        #

        agg_expanded_query = '#weight({queries} {anchor_texts})'.format(
            queries='\n'.join(
                '{weight} {expanded_query}'.format(
                    weight=1.0, expanded_query=expanded_query)
                for expanded_query in expanded_queries),
            anchor_texts='\n'.join(
                '{weight:f} {anchor_text}'.format(
                    weight=self.anchor_texts_beta * weight,
                    anchor_text=anchor_text)
                for weight, anchor_text in
                top_anchor_normalized_frequencies_and_texts))

        logger.debug('Nugget expanded Indri query: %s', agg_expanded_query)

        results = index.query(
            agg_expanded_query,
            document_set=map(
                operator.itemgetter(1),
                index.document_ids(candidate_document_ids)))

        logger.debug('Nugget returned %d results.', len(results))

        for int_document_id, score in results:
            ext_document_id, _ = index.document(int_document_id)

            yield ext_document_id, score


class QCMScorer(SessionScorer):

    def __init__(self, *args, **kwargs):
        super(QCMScorer, self).__init__(*args, **kwargs)

        assert self.desc.qcm_scorer_desc

        assert self.desc.qcm_scorer_desc.belief_type in (
            sesh_pb2.AND, sesh_pb2.OR)

        # HACK; for now.
        if not self.desc.qcm_scorer_desc.params.alpha:
            self.desc.qcm_scorer_desc.params.alpha = 2.2
            self.desc.qcm_scorer_desc.params.beta = 1.8
            self.desc.qcm_scorer_desc.params.epsilon = 0.07
            self.desc.qcm_scorer_desc.params.delta = 0.4

            self.desc.qcm_scorer_desc.params.gamma = 1.0

        logging.info('QCMScorer parameters: %s',
                     self.desc.qcm_scorer_desc.params)

    def score_documents_for_session(
            self,
            session,
            candidate_document_ids,
            index, dictionary,
            anchor_texts,
            word_frequency_index,
            logger,
            f_debug_out):
        if not session.queries:
            return

        interaction_queries = []

        for transition_idx, (previous_interaction, next_interaction) in \
                enumerate(session.iter_transitions(prepend_start=True)):
            previous_serp = previous_interaction.serp
            previous_serp_lm = \
                previous_interaction.create_snippet_lm(dictionary)

            previous_query = previous_interaction.query
            previous_query_bow = dictionary.doc2bow(previous_query)

            next_query = next_interaction.query
            next_query_bow = dictionary.doc2bow(next_query)

            previous_serp_according_to_relevance = sorted(
                ((document.doc_id,
                  previous_serp_lm.score_entity_for_query(
                      document.doc_id, previous_query_bow))
                 for document in previous_serp),
                key=operator.itemgetter(1),
                reverse=True)

            if not previous_serp_according_to_relevance:
                logger.warning('Unable to rank SERP for query "%s".',
                               previous_query)

            logger.debug('Re-ranked SERP for "%s": %s',
                         previous_query,
                         previous_serp_according_to_relevance)

            max_reward_document_id, _ = (
                previous_serp_according_to_relevance[0]
                if previous_serp_according_to_relevance else (None, -np.inf))

            logger.debug('Maximum reward document: %s',
                         max_reward_document_id)

            if not next_query_bow:
                logger.warning('Query %s seems to be OoV.', next_query)

            added_term_ids, removed_term_ids, preserved_term_ids = \
                domain.compute_query_change(previous_interaction,
                                            next_interaction,
                                            dictionary)

            theme_term_ids = \
                domain.compute_theme_terms(previous_interaction,
                                           next_interaction,
                                           dictionary)

            all_term_ids = added_term_ids | \
                removed_term_ids | \
                preserved_term_ids | \
                set(theme_term_ids)

            logger.debug('Transition with theme terms %s, '
                         'added terms %s and removed terms %s.',
                         theme_term_ids, added_term_ids, removed_term_ids)

            query_terms_and_weights = []

            for term_id in all_term_ids:
                term_weight = 0.0

                if term_id in added_term_ids or term_id in preserved_term_ids:
                    logger.debug('Term %d: %.2f + 1.0 (present)',
                                 term_id, term_weight)

                    term_weight += 1.0

                if max_reward_document_id is not None:
                    if term_id in theme_term_ids:
                        _theme_term_weight = (
                            self.desc.qcm_scorer_desc.params.alpha * (
                                1.0 -
                                previous_serp_lm.term_frequency_for_entity(
                                    max_reward_document_id, term_id)))

                        logger.debug('Term %d: %.2f + %.2f (theme term)',
                                     term_id, term_weight, _theme_term_weight)

                        term_weight = term_weight + _theme_term_weight

                    if term_id in added_term_ids:
                        term_max_reward_prob = \
                            previous_serp_lm.term_frequency_for_entity(
                                max_reward_document_id, term_id)

                        if term_max_reward_prob == 0.0:
                            # Term was not present in the maximum reward
                            # document of the previous SERP.
                            term_idf = np.log(index.document_count()) - \
                                np.log(dictionary.dfs[term_id])

                            assert np.isfinite(term_idf)

                            _added_term_not_present_mr = (
                                self.desc.qcm_scorer_desc.params.epsilon *
                                term_idf)

                            logger.debug('Term %d (idf=%.2f): %.2f + %.2f '
                                         '(added term, not max reward)',
                                         term_id, term_idf,
                                         term_weight,
                                         _added_term_not_present_mr)

                            term_weight = (
                                term_weight + _added_term_not_present_mr)
                        else:
                            _added_term_present_mr = (
                                self.desc.qcm_scorer_desc.params.beta *
                                term_max_reward_prob)

                            logger.debug('Term %d: %.2f - %.2f '
                                         '(added term, max reward)',
                                         term_id,
                                         term_weight,
                                         _added_term_present_mr)

                            # Term present in maximum reward document of
                            # previous SERP.
                            term_weight = term_weight - _added_term_present_mr

                    if term_id in removed_term_ids:
                        _removed_term = (
                            self.desc.qcm_scorer_desc.params.delta *
                            previous_serp_lm.term_frequency_for_entity(
                                max_reward_document_id, term_id))

                        logger.debug('Term %d: %.2f - %.2f (removed term)',
                                     term_id,
                                     term_weight,
                                     _removed_term)

                        term_weight = term_weight - _removed_term

                if term_weight != 0.0:
                    query_terms_and_weights.append('{weight} {term}'.format(
                        weight=term_weight, term=dictionary[term_id]))

            if query_terms_and_weights:
                interaction_queries.append('#weight({})'.format(
                    ' '.join(query_terms_and_weights)))
            else:
                interaction_queries.append(None)

        if not filter(lambda query: query is not None, interaction_queries):
            return

        session_query = '#weight({})'.format(
            ' '.join('{interaction_weight} {interaction_query}'.format(
                     interaction_weight=np.power(
                         self.desc.qcm_scorer_desc.params.gamma,
                         len(interaction_queries) - idx),
                     interaction_query=query) if query is not None else ''
                     for idx, query in enumerate(interaction_queries)))

        f_debug_out.write(session_query)
        f_debug_out.write('\n')

        logger.debug('QCM expanded Indri query: %s', session_query)

        results = index.query(
            session_query,
            document_set=map(
                operator.itemgetter(1),
                index.document_ids(candidate_document_ids)))

        logger.debug('QCM returned %d results.', len(results))

        for int_document_id, score in results:
            ext_document_id, _ = index.document(int_document_id)

            yield ext_document_id, score


class IdentityScorer(SessionScorer):

    def __init__(self, *args, **kwargs):
        super(IdentityScorer, self).__init__(*args, **kwargs)

    def score_documents_for_session(
            self,
            session,
            candidate_document_ids,
            index, dictionary,
            anchor_texts,
            word_frequency_index,
            logger,
            f_debug_out):
        for rank, document_id in enumerate(candidate_document_ids):
            yield document_id, -np.log(rank + 1)


SESSION_SCORERS = {
    'oracle': QrelOracleScorer,
    'term_weighting_oracle': TermWeightingOracleScorer,
    'indri': IndriScorer,
    'nugget': NuggetScorer,
    'qcm': QCMScorer,
    'identity': IdentityScorer,
}
