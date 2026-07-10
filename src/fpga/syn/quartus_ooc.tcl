# =====================================================================
# Quartus Prime (Pro) synth + fit for one replication unit.
# Target: Intel/Altera Agilex 5 (also Cyclone/Arria/Stratix with a part swap).
#
# Usage:
#   quartus_sh -t quartus_ooc.tcl <TOP> <PART> <PERIOD_ns> "<GEN=val GEN=val>"
# Example:
#   quartus_sh -t quartus_ooc.tcl tm_map_engine A5ED065BB32AE6SR0 2.0 "GROUP_UNROLL=8"
#
# After it runs, read Fitter/Timing reports (or quartus_sta) for Fmax + ALM/reg use.
# =====================================================================
load_package flow

set TOP      [lindex $quartus(args) 0]
set PART     [lindex $quartus(args) 1]
set PERIOD   [lindex $quartus(args) 2]
set GENERICS [expr {[llength $quartus(args)] > 3 ? [lindex $quartus(args) 3] : ""}]

set SRCDIR [file normalize [file dirname [info script]]/..]
set PROJ   "${TOP}_q"

project_new $PROJ -overwrite
set_global_assignment -name FAMILY "Agilex 5"
set_global_assignment -name DEVICE $PART
set_global_assignment -name TOP_LEVEL_ENTITY $TOP
set_global_assignment -name VHDL_INPUT_VERSION VHDL_2008

# sources from the shared manifest
set fh [open "$SRCDIR/syn/sources.txt" r]
foreach line [split [read $fh] "\n"] {
    set f [string trim $line]
    if {$f eq "" || [string match "#*" $f]} continue
    set_global_assignment -name VHDL_FILE "$SRCDIR/$f"
}
close $fh

# generic overrides: GEN=val -> set_parameter
foreach g $GENERICS {
    set kv [split $g "="]
    set_parameter -name [lindex $kv 0] [lindex $kv 1]
}

# write SDC with the requested period
set sdc [open "$PROJ.sdc" w]
puts $sdc "create_clock -name clk -period $PERIOD \[get_ports clk\]"
puts $sdc "derive_clock_uncertainty"
puts $sdc "set_false_path -from \[get_ports rst\]"
close $sdc
set_global_assignment -name SDC_FILE "$PROJ.sdc"

execute_module -tool map
execute_module -tool fit
execute_module -tool sta
project_close
puts "==== $TOP @ $PART done: read ${PROJ}.fit.rpt (ALM/reg) + ${PROJ}.sta.rpt (Fmax) ===="
