// Configuration file of Table 3 in
//
// C. Van Gysel, E. Kanoulas and M. de Rijke. 2016.
// Lexical Query Modeling in Session Search.

modifier {
    max_session_unique_query_terms: 7
}

scorer: { type: 'oracle' }
scorer: {
    type: 'term_weighting_oracle'

    term_reweighting_oracle_scorer_desc {
            approximation_method: BRUTE
    }
}
scorer: {
    identifier: 'indri_all'
    type: 'indri'

    indri_scorer_desc {
            query_position : ALL
    }
}

candidate_generator: LEMUR
top_candidate_limit: 2000
