-- ============================================================================
-- ORTHON SQL Engine: 06_causal_mechanics/002_role_assignment.sql
-- ============================================================================
-- Assign causal roles to each signal based on lead/lag relationships
--
-- Roles:
--   SOURCE  - Leads others, not led by anyone (driver)
--   SINK    - Led by others, doesn't lead anyone (endpoint)
--   CONDUIT - Both leads and is led (transmitter)
--   ISOLATED - No significant causal relationships
-- ============================================================================

-- Count how many signals each signal leads
CREATE OR REPLACE VIEW v_leads_count AS
SELECT
    leader AS signal_id,
    COUNT(DISTINCT follower) AS n_leads,
    ARRAY_AGG(DISTINCT follower ORDER BY follower) AS leads_signals
FROM v_lead_lag
WHERE leader IS NOT NULL
  AND relationship_strength IN ('very_strong', 'strong', 'moderate')
GROUP BY leader;

-- Count how many signals lead each signal
CREATE OR REPLACE VIEW v_led_by_count AS
SELECT
    follower AS signal_id,
    COUNT(DISTINCT leader) AS n_led_by,
    ARRAY_AGG(DISTINCT leader ORDER BY leader) AS led_by_signals
FROM v_lead_lag
WHERE follower IS NOT NULL
  AND relationship_strength IN ('very_strong', 'strong', 'moderate')
GROUP BY follower;

-- Assign causal roles
CREATE OR REPLACE VIEW v_causal_roles AS
SELECT
    s.signal_id,
    s.signal_class,
    COALESCE(lc.n_leads, 0) AS n_leads,
    COALESCE(lb.n_led_by, 0) AS n_led_by,
    lc.leads_signals,
    lb.led_by_signals,
    -- Role assignment
    CASE
        WHEN COALESCE(lc.n_leads, 0) > 0 AND COALESCE(lb.n_led_by, 0) = 0 THEN 'SOURCE'
        WHEN COALESCE(lc.n_leads, 0) = 0 AND COALESCE(lb.n_led_by, 0) > 0 THEN 'SINK'
        WHEN COALESCE(lc.n_leads, 0) > 0 AND COALESCE(lb.n_led_by, 0) > 0 THEN 'CONDUIT'
        ELSE 'ISOLATED'
    END AS causal_role,
    -- Influence score (positive = net driver, negative = net driven)
    COALESCE(lc.n_leads, 0) - COALESCE(lb.n_led_by, 0) AS influence_score
FROM v_signal_class s
LEFT JOIN v_leads_count lc USING (signal_id)
LEFT JOIN v_led_by_count lb USING (signal_id);

-- Summary by role
CREATE OR REPLACE VIEW v_causal_role_summary AS
SELECT
    causal_role,
    COUNT(*) AS n_signals,
    ARRAY_AGG(signal_id ORDER BY influence_score DESC) AS signals
FROM v_causal_roles
GROUP BY causal_role
ORDER BY
    CASE causal_role
        WHEN 'SOURCE' THEN 1
        WHEN 'CONDUIT' THEN 2
        WHEN 'SINK' THEN 3
        ELSE 4
    END;
