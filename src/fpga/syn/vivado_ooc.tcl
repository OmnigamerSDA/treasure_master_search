# =====================================================================
# Vivado out-of-context synth + place + route for one replication unit.
# Covers Xilinx UltraScale+, Kintex-7, Spartan-7 (and Artix/Zynq).
# NOTE: Spartan-6 is NOT supported by Vivado -> use ISE (see syn/README.md).
#
# Usage (batch):
#   vivado -mode batch -source vivado_ooc.tcl -tclargs <TOP> <PART> <PERIOD_ns> "<GENERICS>"
# Examples:
#   ... -tclargs tm_map_engine   xcku5p-ffvb676-2-e 2.0 "GROUP_UNROLL=8"
#   ... -tclargs tm_map_engine_w xc7k160t-ffg676-2  3.0 "LANES=8 GROUP_UNROLL=4"
#   ... -tclargs tm_map_pipeline xcku5p-ffvb676-2-e 2.5 "N_STAGES=27 GROUP_UNROLL=4"
#
# Reports: <TOP>_<PART>_util.rpt and _timing.rpt + WNS/Fmax to stdout.
# =====================================================================
set TOP     [lindex $argv 0]
set PART    [lindex $argv 1]
set PERIOD  [lindex $argv 2]
set GENERICS [expr {$argc > 3 ? [lindex $argv 3] : ""}]

set SRCDIR [file normalize [file dirname [info script]]/..]
set OUT    "${TOP}_${PART}"

# --- read vendor-neutral VHDL-2008 sources (manifest, minus comments) ---
set fh [open "$SRCDIR/syn/sources.txt" r]
foreach line [split [read $fh] "\n"] {
    set f [string trim $line]
    if {$f eq "" || [string match "#*" $f]} continue
    read_vhdl -vhdl2008 "$SRCDIR/$f"
}
close $fh

# --- generic overrides ---
set gargs {}
foreach g $GENERICS { lappend gargs -generic $g }

# --- OOC synthesis ---
synth_design -top $TOP -part $PART -mode out_of_context {*}$gargs

# clock constraint after synth (ports exist now)
create_clock -name clk -period $PERIOD [get_ports clk]
set_false_path -from [get_ports rst]

opt_design
place_design
route_design

report_utilization      -file "${OUT}_util.rpt"
report_timing_summary   -file "${OUT}_timing.rpt"

# headline numbers to stdout
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set achieved [expr {1000.0 / ($PERIOD - $wns)}]
puts "==== $TOP @ $PART : target ${PERIOD}ns  WNS=${wns}ns  Fmax~[format %.1f $achieved] MHz ===="
puts "==== see ${OUT}_util.rpt for LUT/FF/CARRY/BRAM/DSP ===="
