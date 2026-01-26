-- ============================================================================
-- ORTHON SQL Engine: 06_causal_mechanics/003_causal_chain.sql
-- ============================================================================
-- Construct causal chains showing information flow through the system
--
-- A causal chain is: SOURCE → CONDUIT → ... → CONDUIT → SINK
-- ============================================================================

-- Direct causal edges (leader → follower)
CREATE OR REPLACE VIEW v_causal_edges AS
SELECT
    leader AS from_signal,
    follower AS to_signal,
    lag_magnitude AS lag,
    max_correlation AS strength,
    relationship_strength AS strength_class
FROM v_lead_lag
WHERE leader IS NOT NULL
  AND relationship_strength IN ('very_strong', 'strong', 'moderate');

-- Causal paths (up to 3 hops)
CREATE OR REPLACE VIEW v_causal_paths AS
-- Direct paths (1 hop)
SELECT
    from_signal AS source,
    to_signal AS target,
    1 AS hops,
    from_signal || ' → ' || to_signal AS path,
    strength AS total_strength,
    lag AS total_lag
FROM v_causal_edges

UNION ALL

-- 2-hop paths
SELECT
    e1.from_signal AS source,
    e2.to_signal AS target,
    2 AS hops,
    e1.from_signal || ' → ' || e1.to_signal || ' → ' || e2.to_signal AS path,
    e1.strength * e2.strength AS total_strength,
    e1.lag + e2.lag AS total_lag
FROM v_causal_edges e1
JOIN v_causal_edges e2 ON e1.to_signal = e2.from_signal
WHERE e1.from_signal != e2.to_signal

UNION ALL

-- 3-hop paths
SELECT
    e1.from_signal AS source,
    e3.to_signal AS target,
    3 AS hops,
    e1.from_signal || ' → ' || e1.to_signal || ' → ' || e2.to_signal || ' → ' || e3.to_signal AS path,
    e1.strength * e2.strength * e3.strength AS total_strength,
    e1.lag + e2.lag + e3.lag AS total_lag
FROM v_causal_edges e1
JOIN v_causal_edges e2 ON e1.to_signal = e2.from_signal
JOIN v_causal_edges e3 ON e2.to_signal = e3.from_signal
WHERE e1.from_signal != e2.to_signal
  AND e1.from_signal != e3.to_signal
  AND e2.from_signal != e3.to_signal;

-- Best path between each source-target pair
CREATE OR REPLACE VIEW v_causal_chains AS
WITH ranked AS (
    SELECT
        source,
        target,
        hops,
        path,
        total_strength,
        total_lag,
        ROW_NUMBER() OVER (
            PARTITION BY source, target
            ORDER BY total_strength DESC
        ) AS rn
    FROM v_causal_paths
    WHERE source IN (SELECT signal_id FROM v_causal_roles WHERE causal_role = 'SOURCE')
      AND target IN (SELECT signal_id FROM v_causal_roles WHERE causal_role IN ('SINK', 'CONDUIT'))
)
SELECT
    source,
    target,
    hops,
    path,
    ROUND(total_strength, 4) AS chain_strength,
    total_lag AS chain_lag
FROM ranked
WHERE rn = 1
ORDER BY chain_strength DESC;

-- System causal summary
CREATE OR REPLACE VIEW v_causal_summary AS
SELECT
    (SELECT COUNT(*) FROM v_causal_edges) AS n_causal_edges,
    (SELECT COUNT(DISTINCT from_signal) FROM v_causal_edges) AS n_drivers,
    (SELECT COUNT(DISTINCT to_signal) FROM v_causal_edges) AS n_driven,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'SOURCE') AS n_sources,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'SINK') AS n_sinks,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'CONDUIT') AS n_conduits,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'ISOLATED') AS n_isolated;
