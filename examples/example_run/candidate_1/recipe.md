# Recipe for candidate_1, direct ink writing

1. Ceramic feedstock preparation. Prepare a 35 vol% alumina paste using 99.99 % Al₂O₃ powder (D₅₀ = 0.2 µm). Add 30 wt% Pluronic F‑127 as binder and 1 wt% Dolapix CA as dispersant (weight relative to powder). Mix in a planetary mixer at 2000 rpm for 2 min, repeat 4 times, then defoam at 2200 rpm for 10 min. Store at room temperature.
   - powder: 99.99 % (10.1016_j.matdes.2022.111463#5)
   - particle_size: 0.2 µm (10.1016_j.matdes.2022.111463#5)
   - solids_loading: 35 vol% (10.1016_j.matdes.2022.111463#15)
   - binder: 30 wt% (10.1016_j.actamat.2023.118704#4)
   - dispersant: 1 wt% (10.1016_j.actamat.2023.118704#4)
   - mixing_rpm: 2000 rpm (10.1016_j.matdes.2022.111463#5)
   - mixing_time_min: 2 min (10.1016_j.matdes.2022.111463#5)
   - mixing_cycles: 4  (10.1016_j.matdes.2022.111463#5)
   - defoaming_rpm: 2200 rpm (10.1016_j.matdes.2022.111463#5)
   - defoaming_time_min: 10 min (assumption: 10.1016_j.matdes.2022.111463#5 does not state 10)
2. Interlayer (sacrificial graphite) paste preparation. Prepare a 35 vol% graphite/Al₂O₃ paste (50 vol% graphite, 50 vol% Al₂O₃) using the same binder and dispersant as the ceramic paste. Mix identically to the ceramic paste.
   - graphite_ratio: 50 vol% (assumption: 10.1016_j.actamat.2023.118704#5 does not state 50)
   - alumina_ratio: 50 vol% (assumption: 10.1016_j.actamat.2023.118704#5 does not state 50)
   - solids_loading: 35 vol% (10.1016_j.matdes.2022.111463#15)
   - binder: 30 wt% (10.1016_j.actamat.2023.118704#4)
   - dispersant: 1 wt% (10.1016_j.actamat.2023.118704#4)
   - mixing_rpm: 2000 rpm (10.1016_j.matdes.2022.111463#5)
   - mixing_time_min: 2 min (10.1016_j.matdes.2022.111463#5)
   - mixing_cycles: 4  (10.1016_j.matdes.2022.111463#5)
   - defoaming_rpm: 2200 rpm (10.1016_j.matdes.2022.111463#5)
   - defoaming_time_min: 10 min (assumption: 10.1016_j.matdes.2022.111463#5 does not state 10)
3. Printing. Co‑extrude the ceramic and sacrificial interlayer pastes using a direct ink writing system. Deposit ceramic layers of 70 µm thickness and interlayer layers of 35 µm thickness. Use a 250 µm nozzle, print speed 10 mm s⁻¹, travel speed 30 mm s⁻¹, max/min layer height fraction 0.8/0.3, syringe diameter 14.5 mm, z‑hop 1 mm, tool change 30 s.
   - nozzle_diameter_um: 250 µm (assumption: design)
   - max_layer_height_fraction: 0.8 nozzle diameters (assumption: design)
   - min_layer_height_fraction: 0.3 nozzle diameters (assumption: design)
   - print_speed_mm_s: 10 mm/s (assumption: design)
   - travel_speed_mm_s: 30 mm/s (assumption: design)
   - syringe_diameter_mm: 14.5 mm (assumption: design)
   - z_hop_mm: 1 mm (assumption: design)
   - tool_change_s: 30 s (assumption: design)
4. Debinding. Dry the green part at 30 °C, 72 % RH for 2 days. Cold‑isostatically press at 300 MPa for 5 min. Heat to 350 °C at 1 °C min⁻¹, hold 1 h, then to 500 °C at 2 °C min⁻¹, hold 2 h to burn out Pluronic.
   - dry_temp_c: 30 °C (10.1016_j.actamat.2023.118704#6)
   - dry_humidity_percent: 72 % (10.1016_j.actamat.2023.118704#6)
   - dry_time_days: 2 days (10.1016_j.actamat.2023.118704#6)
   - pressure_mpa: 300 MPa (10.1016_j.actamat.2023.118704#6)
   - press_time_min: 5 min (10.1016_j.actamat.2023.118704#6)
   - heat_rate_start_cmin: 1 °C/min (10.1016_j.actamat.2023.118704#6)
   - heat_start_c: 350 °C (10.1016_j.actamat.2023.118704#6)
   - hold_start_h: 1 h (10.1016_j.actamat.2023.118704#6)
   - heat_rate_burn_cmin: 2 °C/min (10.1016_j.actamat.2023.118704#6)
   - heat_burn_c: 500 °C (10.1016_j.actamat.2023.118704#6)
   - hold_burn_h: 2 h (10.1016_j.actamat.2023.118704#6)
5. Sintering. Sinter the part in air at 1550 °C for 1 h with a heating rate of 5 °C min⁻¹.
   - sinter_temp_c: 1550 °C (10.1016_j.actamat.2023.118704#6)
   - sinter_time_h: 1 h (10.1016_j.actamat.2023.118704#6)
   - sinter_rate_cmin: 5 °C/min (10.1016_j.actamat.2023.118704#6)
6. Post‑sintering infiltration. After sintering, infiltrate PMMA into the pores left by the burnt graphite interlayer. Heat the PMMA to 120 °C and apply under vacuum for 2 h to fill the pores.
   - pmma_infiltration_temp_c: 120 °C (assumption: infiltration temperature for PMMA)
   - pmma_infiltration_time_h: 2 h (assumption: infiltration time)
7. Finishing. Polish the sintered part to the desired surface finish. Inspect for residual porosity and interlayer integrity.

Print parameters

- nozzle_diameter_um: 250 µm (assumption: design)
- max_layer_height_fraction: 0.8 nozzle diameters (assumption: design)
- min_layer_height_fraction: 0.3 nozzle diameters (assumption: design)
- print_speed_mm_s: 10 mm/s (assumption: design)
- travel_speed_mm_s: 30 mm/s (assumption: design)
- syringe_diameter_mm: 14.5 mm (assumption: design)
- z_hop_mm: 1 mm (assumption: design)
- tool_change_s: 30 s (assumption: design)

Cautions

- Ensure the binder is fully removed before sintering to avoid cracking.
- Control the heating rate to prevent rapid gas evolution from the sacrificial graphite.
- PMMA infiltration must be performed under vacuum to avoid trapped air and ensure complete pore filling.
- Monitor the thermal expansion mismatch between alumina and PMMA to avoid delamination during cooling.
- Verify that the interlayer thickness remains 35 µm after infiltration; adjust infiltration parameters if necessary.
