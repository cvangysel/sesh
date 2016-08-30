// Configuration file of Table 2, Figure 1 and Figure 2 in
//
// C. Van Gysel, E. Kanoulas and M. de Rijke. 2016.
// Lexical Query Modeling in Session Search.

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
