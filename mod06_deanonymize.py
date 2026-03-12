import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_cols = ["age", "zip3", "gender"]

    # Keep keys that appear exactly once in each dataset for one-to-one linkage.
    anon_unique_keys = (
        anon_df.groupby(quasi_cols)
        .size()
        .reset_index(name="anon_count")
        .query("anon_count == 1")
        .drop(columns=["anon_count"])
    )
    aux_unique_keys = (
        aux_df.groupby(quasi_cols)
        .size()
        .reset_index(name="aux_count")
        .query("aux_count == 1")
        .drop(columns=["aux_count"])
    )

    valid_keys = anon_unique_keys.merge(aux_unique_keys, on=quasi_cols, how="inner")

    anon_unique = anon_df.merge(valid_keys, on=quasi_cols, how="inner")
    aux_unique = aux_df.merge(valid_keys, on=quasi_cols, how="inner")

    matches = anon_unique.merge(aux_unique, on=quasi_cols, how="inner")

    return matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return len(matches_df) / len(anon_df)
