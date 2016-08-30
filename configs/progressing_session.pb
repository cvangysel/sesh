// Configuration file of Figure 3a in
//
// C. Van Gysel, E. Kanoulas and M. de Rijke. 2016.
// Lexical Query Modeling in Session Search.

modifier {
    identifier: 'position_1'

    min_session_length: 5
    max_session_length: 5

    session_cutoff: 4
}
modifier {
    identifier: 'position_2'

    min_session_length: 5
    max_session_length: 5

    session_cutoff: 3
}
modifier {
    identifier: 'position_3'

    min_session_length: 5
    max_session_length: 5

    session_cutoff: 2
}
modifier {
    identifier: 'position_4'

    min_session_length: 5
    max_session_length: 5

    session_cutoff: 1
}
modifier {
    identifier: 'position_5'

    min_session_length: 5
    max_session_length: 5

    session_cutoff: 0
}

scorer: {
    identifier: 'qcm'
    type: 'qcm'

    qcm_scorer_desc {
        belief_type: AND
    }
}
scorer: {
    identifier: 'nugget_RL2'
    type: 'nugget'

    nugget_scorer_desc {
        reading_level: RL2
    }
}
scorer: {
    identifier: 'nugget_RL3'
    type: 'nugget'

    nugget_scorer_desc {
        reading_level: RL3
    }
}
scorer: {
    identifier: 'nugget_RL4'
    type: 'nugget'

    nugget_scorer_desc {
        reading_level: RL4
    }
}
scorer: { type: 'oracle' }
scorer: {
    identifier: 'indri_all'
    type: 'indri'

    indri_scorer_desc {
            query_position : ALL
    }
}
scorer: {
    identifier: 'indri_first'
    type: 'indri'

    indri_scorer_desc {
            query_position : FIRST
    }
}
scorer: {
    identifier: 'indri_last'
    type: 'indri'

    indri_scorer_desc {
            query_position: LAST
    }
}

candidate_generator: LEMUR
top_candidate_limit: 2000
