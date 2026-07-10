# Xilinx (Vivado) clock constraint for OOC synthesis.
# Override the period to sweep Fmax: vivado ... -tclargs PERIOD=<ns>.
# Default 2.5 ns = 400 MHz target (raise/lower to find the achievable Fmax).
create_clock -name clk -period 2.500 [get_ports clk]

# OOC: treat all non-clock ports as set by an ideal source/dest so timing
# reports reflect the internal logic, not unconstrained I/O.
set_input_delay  -clock clk 0.000 [all_inputs]
set_output_delay -clock clk 0.000 [all_outputs]
set_false_path -from [get_ports rst]
