"""Reviewed ambiguity metadata applied after sentence segmentation."""

BOUNDARY_CHALLENGES = {
    31297: {
        "flag": "ambiguous_ellipsis_boundary",
        "review_status": "unresolved",
        "expectation": "preserve_v2_segmentation_and_flag_for_human_review",
    },
    39538: {
        "flag": "ambiguous_quote_attribution",
        "review_status": "unresolved",
        "expectation": "preserve_v2_segmentation_and_flag_for_human_review",
    },
}
