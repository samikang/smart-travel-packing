"""
Editable Packing List UI
=========================
Renders an editable dataframe for the draft packing list (Section 4).
Users can adjust quantities, uncheck items to remove them, and add
new custom items not suggested by the ML recommender.
"""

import streamlit as st
import pandas as pd


def render_packing_editor(
    items_data: list,
    title: str = "📋 Draft Packing List",
) -> list:
    """
    Renders an editable dataframe packing list.

    Args:
        items_data: List of dicts. Required keys: 'name'.
                    Optional: 'category', 'quantity', 'keep',
                              'weight_g', 'volume_l'.
    Returns:
        List of item dicts the user chose to keep (keep=True, qty > 0).
    """
    if not items_data:
        st.info("No items generated yet. Complete slot detection and image upload first.")
        return []

    st.subheader(title)

    df = pd.DataFrame(items_data)

    # Ensure all expected columns exist with sensible defaults
    df["quantity"] = df.get("quantity", pd.Series([1] * len(df))).fillna(1).astype(int)
    df["keep"]     = df.get("keep",     pd.Series([True] * len(df))).fillna(True)
    df["weight_g"] = df.get("weight_g", pd.Series([0] * len(df))).fillna(0).astype(int)
    df["volume_l"] = df.get("volume_l", pd.Series([0.0] * len(df))).fillna(0.0)
    df["category"] = df.get("category", pd.Series([""] * len(df))).fillna("")

    edited_df = st.data_editor(
        df[["name", "category", "quantity", "keep", "weight_g", "volume_l"]],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",  # allows adding new rows
        column_config={
            "name":     st.column_config.TextColumn(
                "Item", help="Item name — edit to rename or add custom items."
            ),
            "category": st.column_config.TextColumn(
                "Type", disabled=True,
                help="Clothing | Gear | Your Wardrobe",
            ),
            "quantity": st.column_config.NumberColumn(
                "Qty", min_value=0, max_value=20, step=1,
                help="Set to 0 or uncheck to exclude from optimisation.",
            ),
            "keep":     st.column_config.CheckboxColumn(
                "Pack?", default=True,
                help="Uncheck to exclude this item from the final packing list.",
            ),
            "weight_g": st.column_config.NumberColumn(
                "Weight (g)", disabled=True,
                help="Estimated packed weight in grams.",
            ),
            "volume_l": st.column_config.NumberColumn(
                "Volume (L)", format="%.1f", disabled=True,
                help="Estimated packed volume in litres.",
            ),
        },
        key="packing_editor",
    )

    # Summary totals below the table
    kept = edited_df[edited_df["keep"] & (edited_df["quantity"] > 0)]
    total_w = kept["weight_g"].sum() / 1000
    total_v = kept["volume_l"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Items to Pack",   len(kept))
    c2.metric("Total Weight",    f"{total_w:.1f} kg")
    c3.metric("Total Volume",    f"{total_v:.1f} L")

    return kept.to_dict("records")
