# Generic SDC clock constraint (Quartus / Libero / Radiant) for OOC synthesis.
# Edit the period to sweep Fmax. 2.5 ns = 400 MHz target.
create_clock -name clk -period 2.500 [get_ports clk]
derive_clock_uncertainty

# Relax I/O so reports reflect internal logic (OOC). Tools differ slightly;
# the false_path on reset is the portable part.
set_false_path -from [get_ports rst]
