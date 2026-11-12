# sky130 time unit: ns
current_design tc_fda_top

set clk_name  core_clock
set clk_port  clk
set clk_period [expr {[info exists ::env(TC_CLK_PERIOD)] ? $::env(TC_CLK_PERIOD) : 10.0}]
set io_frac   0.2

create_clock -name $clk_name -period $clk_period [get_ports $clk_port]

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] [get_ports $clk_port]]
set_input_delay  [expr {$clk_period * $io_frac}] -clock $clk_name $non_clock_inputs
set_output_delay [expr {$clk_period * $io_frac}] -clock $clk_name [all_outputs]
