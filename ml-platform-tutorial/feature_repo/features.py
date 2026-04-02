"""
Feast feature definitions for fraud detection.

This file defines the core Feast components:
- Entities: The lookup keys used to retrieve features (e.g., merchant_category)
- Data Sources: Where raw feature data is stored (Parquet files)
- Feature Views: The actual features and their schemas

Why this matters: These definitions serve as the SINGLE SOURCE OF TRUTH.
Both training (offline) and serving (online) use identical feature definitions,
preventing the infamous "training-serving skew" problem where models perform
well in training but poorly in production due to feature inconsistencies.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType

from feast.types import Float32, Int64

# Entity: The primary key used to look up features in the feature store.
# In fraud detection, we lookup features by merchant_category (e.g., 'online', 'grocery').
# This enables fast feature retrieval at inference time without recalculating aggregates.
merchant = Entity(
    name="merchant_category",
    description="Merchant category for the transaction (e.g., 'online', 'grocery')",
    value_type=ValueType.STRING,
)

# FileSource: Defines where the raw feature data lives.
# We use a Parquet file containing pre-computed merchant statistics.
# The path is relative to the feature_repo directory.
# The timestamp_field tells Feast how to handle data freshness and TTL.
merchant_stats_source = FileSource(
    name="merchant_stats_source",
    path="data/merchant_features.parquet",
    timestamp_field="event_timestamp",
)

# FeatureView: Groups related features together with their schema and configuration.
# These features are computed per merchant_category from historical transaction data:
# - avg_amount: Average transaction amount for this merchant type
# - transaction_count: Total number of transactions seen
# - fraud_rate: Historical fraud rate (critical for fraud detection!)
#
# TTL (Time To Live): 7 days - features older than this are refreshed from source.
# online=True: Enables low-latency serving from the online feature store.
merchant_stats_fv = FeatureView(
    name="merchant_stats",
    description="Aggregated statistics per merchant category",
    entities=[merchant],
    ttl=timedelta(days=7),
    schema=[
        Field(name="avg_amount", dtype=Float32, description="Average transaction amount"),
        Field(name="transaction_count", dtype=Int64, description="Number of transactions"),
        Field(name="fraud_rate", dtype=Float32, description="Historical fraud rate"),
    ],
    source=merchant_stats_source,
    online=True,
)