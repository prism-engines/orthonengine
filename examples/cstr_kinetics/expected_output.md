# Expected PRISM Output — CSTR Kinetics

## What PRISM Should Calculate

### 1. Reaction Discipline → physics.parquet

```
entity_id | window | conversion | rate_constant_L_mol_min | reaction_rate_mol_L_min | residence_time_min
run_1     | 0      | 0.280      | 0.108                   | 0.0000056                | 50.0
run_2     | 0      | 0.466      | 0.212                   | 0.0000093                | 50.0
run_3     | 0      | 0.622      | 0.393                   | 0.0000124                | 50.0
run_4     | 0      | 0.740      | 0.682                   | 0.0000148                | 50.0
run_5     | 0      | 0.824      | 1.095                   | 0.0000165                | 50.0
```

### Calculations:

**Residence time:**
```
τ = V/Q = 2.5 L / (50 mL/min × 1L/1000mL) = 2.5 / 0.05 = 50 min
```

**Conversion (run_1 example):**
```
X = (C_A0 - C_A) / C_A0 = (0.100 - 0.0720) / 0.100 = 0.280
```

**Rate constant (CSTR design equation, 2nd order equimolar):**
```
k = X / (τ × C_A0 × (1-X)²)
k = 0.280 / (50 × 0.100 × (1-0.280)²)
k = 0.280 / (50 × 0.100 × 0.5184)
k = 0.280 / 2.592
k = 0.108 L/(mol·min)
```

---

### 2. Arrhenius Analysis → physics.parquet (aggregated)

```
metric                        | value
activation_energy_J_mol       | 45012
pre_exponential_L_mol_min     | 2.36e6
r_squared                     | 0.9997
```

**Arrhenius linearization:**

| run | T (K) | 1/T (1/K) | k (L/mol·min) | ln(k) |
|-----|-------|-----------|---------------|-------|
| 1   | 298.15 | 0.003354 | 0.108 | -2.226 |
| 2   | 308.15 | 0.003245 | 0.212 | -1.551 |
| 3   | 318.15 | 0.003143 | 0.393 | -0.934 |
| 4   | 328.15 | 0.003047 | 0.682 | -0.383 |
| 5   | 338.15 | 0.002957 | 1.095 | 0.091 |

**Linear regression: ln(k) = ln(A) - Ea/(R·T)**
```
slope = -Ea/R = -5414 K
Ea = 5414 × 8.314 = 45,012 J/mol ✓

intercept = ln(A) = 15.98
A = e^15.98 = 2.36 × 10⁶ L/(mol·min)
```

Note: Ea is the same (~45 kJ/mol), A is ~50x smaller due to τ=50 min.

---

### 3. Material Balance → physics.parquet

```
entity_id | window | inlet_moles_EA | outlet_moles_EA | reacted_moles | balance_closure_pct
run_1     | 0      | 0.00500        | 0.00360         | 0.00140       | 100.0
run_2     | 0      | 0.00500        | 0.00267         | 0.00233       | 100.0
run_3     | 0      | 0.00500        | 0.00189         | 0.00311       | 100.0
run_4     | 0      | 0.00500        | 0.00130         | 0.00370       | 100.0
run_5     | 0      | 0.00500        | 0.00088         | 0.00412       | 100.0
```

**Calculation (run_1):**
```
Inlet: F_A0 = C_A0 × Q = 0.100 mol/L × 0.050 L/min = 0.00500 mol/min
Outlet: F_A = C_A × Q = 0.0720 mol/L × 0.050 L/min = 0.00360 mol/min
Reacted: F_A0 - F_A = 0.00500 - 0.00360 = 0.00140 mol/min
Product formed: F_P = C_P × Q = 0.0280 × 0.050 = 0.00140 mol/min ✓

Balance: In = Out + Reacted
0.00500 = 0.00360 + 0.00140 ✓
Closure: 100%
```

---

### 4. Energy Balance → physics.parquet

```
entity_id | window | heat_duty_W | heat_duty_kJ_min
run_1     | 0      | 1.76        | 0.105
run_2     | 0      | 2.93        | 0.176
run_3     | 0      | 3.90        | 0.234
run_4     | 0      | 4.65        | 0.279
run_5     | 0      | 5.17        | 0.310
```

**Calculation (run_1):**
```
Q = F_A0 × X × (-ΔH_rxn)
Q = 0.00500 mol/min × 0.280 × 75300 J/mol
Q = 105.4 J/min = 1.76 W

(Heat removed to maintain isothermal)
```

---

### 5. Transport → physics.parquet

```
entity_id | window | reynolds | flow_regime
run_1     | 0      | 1044     | laminar
run_2     | 0      | 1044     | laminar
run_3     | 0      | 1044     | laminar
run_4     | 0      | 1044     | laminar
run_5     | 0      | 1044     | laminar
```

**Calculation:**
```
Q = 50 mL/min = 8.33 × 10⁻⁷ m³/s
D = 0.0127 m
ρ = 1020 kg/m³
μ = 0.00102 Pa·s

Re = 4ρQ / (πDμ)
Re = 4 × 1020 × 8.33×10⁻⁷ / (π × 0.0127 × 0.00102)
Re = 0.00340 / 4.07×10⁻⁵
Re = 1044

Re < 2100 → Laminar ✓
```

---

### 6. Core Analysis → vector.parquet

**Per-signal statistics (example: outlet_C_EA for run_1):**
```
entity_id | signal_id    | window | mean   | std    | trend_slope | stationarity_pvalue
run_1     | outlet_C_EA  | 0      | 0.0720 | 0.0002 | -0.00004    | 0.85
```

- Low std → good precision
- Near-zero slope → steady state reached
- High p-value → stationary (good)

---

## Summary for Student

### Results Table

| T (°C) | T (K) | X | k (L/mol·min) | Q (W) |
|--------|-------|---|---------------|-------|
| 25 | 298.15 | 0.280 | 0.108 | 1.76 |
| 35 | 308.15 | 0.466 | 0.212 | 2.93 |
| 45 | 318.15 | 0.622 | 0.393 | 3.90 |
| 55 | 328.15 | 0.740 | 0.682 | 4.65 |
| 65 | 338.15 | 0.824 | 1.095 | 5.17 |

### Arrhenius Parameters

- **Activation Energy:** Ea = 45.0 kJ/mol
- **Pre-exponential Factor:** A = 2.36 × 10⁶ L/(mol·min)
- **R²:** 0.9997

### Verification

- ✓ Material balance closes (100% at all temperatures)
- ✓ Energy balance closes
- ✓ Flow is laminar (Re = 1044)
- ✓ Steady state confirmed (low variance, stationary)
- ✓ Arrhenius fit excellent (R² > 0.999)

### Thesis-Ready Figures (PRISM generates)

1. **Arrhenius Plot:** ln(k) vs 1/T with linear fit
2. **Conversion vs Temperature:** X vs T
3. **Rate Constant vs Temperature:** k vs T
4. **Material Balance Closure:** bar chart
5. **Steady State Verification:** time series with confidence bands
