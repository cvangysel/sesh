from cvangysel import language_models, nltk_utils, io_utils

import logging
import itertools

from lxml import etree


def create_tokenizer_fn(dictionary):
    def tokenizer_fn(text):
        return [
            token
            for token in io_utils.tokenize_text(text)
            if dictionary.has_token(token)] \
            if text is not None else None

    return tokenizer_fn


class DomainObject(object):

    def __init__(self, **kwargs):
        pass


class Document(DomainObject):

    def __init__(self, doc_id, url, title, body,
                 tokenizer_fn,
                 **kwargs):
        super(Document, self).__init__(**kwargs)

        self.doc_id = doc_id
        self.url = url

        self.title = tokenizer_fn(title)
        self.body = tokenizer_fn(body)

    def __iter__(self):
        return itertools.chain(
            self.title if self.title is not None else [],
            self.body if self.body is not None else [])

    def __eq__(self, other):
        if isinstance(other, Document):
            return (self.doc_id == other.doc_id)
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return self.doc_id.__hash__()


class Click(DomainObject):

    def __init__(self, start_time, end_time, doc_id,
                 **kwargs):
        super(Click, self).__init__(**kwargs)

        self.start_time = start_time
        self.end_time = end_time

        self.doc_id = doc_id


class Interaction(DomainObject):

    def __init__(self, query, serp, clicks, tokenizer_fn,
                 **kwargs):
        super(Interaction, self).__init__(**kwargs)

        self.original_query = query
        self.query = tokenizer_fn(query)

        if not self.query:
            logging.warning('Raw query "%s" translated to empty tokens.',
                            query)

        logging.debug('Raw query "%s" tokenized to "%s".',
                      query, self.query)

        self.serp = serp
        self.clicks = clicks

    def __repr__(self):
        return 'Interaction "{}" with SERP {} and clicks {}'.format(
            self.query, self.serp, self.clicks)

    def create_snippet_lm(self, dictionary, **model_kwargs):
        assert self.serp is not None

        index = language_models.WordFrequencyIndex(
            nltk_utils.UniformProbDist(len(dictionary)))

        snippets = [
            (dictionary.doc2bow(document), document.doc_id)
            for document in self.serp]

        index.initialize(snippets)

        if 'scoring_method' not in model_kwargs:
            model_kwargs['scoring_method'] = \
                language_models.LanguageModel.OR_BELIEF

        if 'smoothing_method' not in model_kwargs:
            model_kwargs['smoothing_method'] = None

        serp_lm = language_models.LanguageModel(
            index, **model_kwargs)

        return serp_lm


def compute_query_change(previous_interaction, next_interaction,
                         dictionary):
    previous_query = set(map(dictionary.translate_token,
                             previous_interaction.query))
    next_query = set(map(dictionary.translate_token,
                         next_interaction.query))

    added_terms = next_query - previous_query
    removed_terms = previous_query - next_query
    preserved_terms = next_query & previous_query

    return added_terms, removed_terms, preserved_terms


def compute_theme_terms(previous_interaction, next_interaction,
                        dictionary):
    previous_query = list(map(dictionary.translate_token,
                              previous_interaction.query))
    next_query = list(map(dictionary.translate_token,
                          next_interaction.query))

    # Adapted from http://rosettacode.org/wiki/
    # Longest_common_subsequence#Recursion_7
    def lcs(first, second):
        if not first or not second:
            return ()

        x, xs, y, ys = first[0], first[1:], second[0], second[1:]

        if x == y:
            return (x,) + lcs(xs, ys)
        else:
            return max(lcs(first, ys), lcs(xs, second), key=len)

    return lcs(previous_query, next_query)


class Session(DomainObject):

    def __init__(self, track_edition, session_id,
                 interactions, current_query,
                 **kwargs):
        super(Session, self).__init__(**kwargs)

        self.track_edition = track_edition
        self.session_id = session_id

        self.interactions = interactions

        assert current_query, \
            'Session {session_id} has no current query.'.format(
                session_id=session_id)

        self.interactions.append(
            Interaction(query=current_query,
                        serp=None,
                        clicks=None,
                        **kwargs))

    @property
    def num_serp_documents(self):
        return sum(
            len(interaction.serp)
            for interaction in self.interactions
            if interaction.serp is not None)

    @property
    def num_serp_clicks(self):
        return sum(
            len(interaction.clicks)
            for interaction in self.interactions
            if interaction.clicks is not None)

    @property
    def queries(self):
        return [interaction.query for interaction in self.interactions]

    @property
    def terms(self):
        return set([term for interaction in self.interactions
                    for term in interaction.query])

    def __str__(self):
        return 'Session {0}'.format(self.session_id)

    def __repr__(self):
        return '{0} ({1} SERP documents, {2} SERP clicks): {3}'.format(
            str(self),
            self.num_serp_documents, self.num_serp_clicks,
            ' -> '.join(map(str, self.queries)))

    def __len__(self):
        return len(self.interactions)

    def iter_transitions(self, reverse=False, prepend_start=False):
        if prepend_start:
            null_interaction = Interaction(
                query='',
                serp=[],
                clicks=[],
                tokenizer_fn=lambda text: [])

            interactions = [null_interaction]

            interactions.extend(self.interactions)
        else:
            interactions = self.interactions

        if reverse:
            range_it = range(len(interactions) - 2, -1, -1)
        else:
            range_it = range(len(interactions) - 1)

        for idx in range_it:
            yield interactions[idx], interactions[idx + 1]

    def iter_interactions(self, reverse=False):
        if reverse:
            s = slice(None, None, -1)
        else:
            s = slice(None, None, 1)

        for interaction in self.interactions[s]:
            yield interaction


class SessionTrackInputFormat(object):

    def document_id_name(self):
        raise NotImplementedError

    def extract_topic_id(self, session_element):
        return int(session_element.find('topic').get('num'))


class SessionTrack2011InputFormat(SessionTrackInputFormat):

    def __init__(self):
        self.topics = {}

    def document_id_name(self):
        return 'clueweb09id'

    def extract_topic_id(self, session_element):
        topic = session_element.find('topic/title').text

        if topic not in self.topics:
            self.topics[topic] = len(self.topics) + 1

        return self.topics[topic]


class SessionTrack2012InputFormat(SessionTrackInputFormat):

    def document_id_name(self):
        return 'clueweb09id'


class SessionTrack2014InputFormat(SessionTrackInputFormat):

    def document_id_name(self):
        return 'clueweb12id'


SESSION2011, SESSION2012, SESSION2013, SESSION2014 = range(2011, 2014 + 1)

INPUT_FORMATS = {
    SESSION2011: SessionTrack2011InputFormat,
    SESSION2012: SessionTrack2012InputFormat,
    SESSION2013: SessionTrack2014InputFormat,
    SESSION2014: SessionTrack2014InputFormat,
}


def construct_sessions(f_session_xml, num_sessions, dictionary,
                       encoding='utf8'):
    xml = f_session_xml.read().encode(f_session_xml.encoding)

    def to_unicode(str_or_unicode):
        if str_or_unicode is None:
            return None

        return str_or_unicode.encode(encoding).decode(encoding)

    documents = set()
    sessions = {}

    session_id_to_topic_id = {}

    tree = etree.fromstring(xml)
    root = tree.getroottree().getroot()

    if root.tag == 'sessiontrack2011':
        track_edition = SESSION2011
    elif root.tag == 'sessiontrack2012':
        track_edition = SESSION2012
    elif root.tag == 'sessiontrack2013':
        track_edition = SESSION2013
    elif root.tag == 'sessiontrack2014':
        track_edition = SESSION2014
    else:
        raise NotImplementedError()

    tokenizer_fn = create_tokenizer_fn(dictionary)

    session_track = INPUT_FORMATS[track_edition]()

    for session in tree.iterfind('session'):
        session_id = int(session.get('num'))
        topic_id = session_track.extract_topic_id(session)

        assert session_id not in session_id_to_topic_id

        session_id_to_topic_id[session_id] = topic_id

        interactions = []

        for interaction in session.iterfind('interaction'):
            query = to_unicode(interaction.find('query').text)

            serp = []

            for result in interaction.iterfind('results/result'):
                url = to_unicode(result.find('url').text)
                doc_id = to_unicode(
                    result.find(session_track.document_id_name()).text)

                title = to_unicode(result.find('title').text)
                body = to_unicode(result.find('snippet').text)

                document = Document(doc_id, url, title, body,
                                    tokenizer_fn=tokenizer_fn)

                serp.append(document)
                documents.add(document)

            clicks = []

            for click in interaction.iterfind('clicked/click'):
                start_time = None  # float(click.get('starttime'))
                end_time = None  # float(click.get('endtime'))

                rank = int(click.find('rank').text)

                if rank > len(serp):
                    logging.warning(
                        'Session %d contains click on '
                        'non-existing SERP document.',
                        session_id)

                    continue

                doc_id = serp[rank - 1].doc_id

                click = Click(start_time, end_time, doc_id,
                              tokenizer_fn=tokenizer_fn)

                clicks.append(click)

            interaction = Interaction(query, serp, clicks,
                                      tokenizer_fn=tokenizer_fn)

            interactions.append(interaction)

        current_query_element = session.find('currentquery/query')
        current_query = to_unicode(current_query_element.text) \
            if current_query_element is not None else None

        if not current_query:
            logging.warning('Session %d has no current query; skipping.',
                            session_id)

            continue

        session = Session(track_edition, session_id,
                          interactions, current_query,
                          tokenizer_fn=tokenizer_fn)

        sessions[session_id] = session

        if num_sessions is not None and \
                len(sessions) >= num_sessions:
            break

    return track_edition, documents, sessions, session_id_to_topic_id


def get_document_set(sessions):
    return set([
        document
        for session in sessions
        for interaction in (
            interaction for interaction in session.interactions
            if interaction.serp)
        for document in interaction.serp])


def alter_sessions(session_id_and_sessions, modifier):
    altered_sessions = {}

    for session_id, session in session_id_and_sessions.items():
        if modifier.max_session_length and \
                len(session) > modifier.max_session_length:
            continue

        if modifier.max_session_length and \
                len(session) < modifier.min_session_length:
            continue

        if modifier.max_session_unique_query_terms and \
                len(session.terms) > modifier.max_session_unique_query_terms:
            continue

        assert modifier.session_cutoff >= 0

        if modifier.session_cutoff:
            if len(session) <= modifier.session_cutoff:
                logging.warning('%s only has %d interactions, while '
                                'the session cut-off is set to %d.',
                                session, len(session),
                                modifier.session_cutoff)

            session.interactions = session.interactions[
                :-modifier.session_cutoff]

        assert modifier.session_context_length >= 0

        if modifier.session_context_length:
            if len(session) < modifier.session_context_length:
                logging.warning('%s only has %d interactions, while the '
                                'context length was fixed at %d.',
                                session, len(session),
                                modifier.session_context_length)

            session.interactions = session.interactions[
                -modifier.session_context_length:]

            assert len(session) <= modifier.session_context_length

        altered_sessions[session_id] = session

    return altered_sessions
