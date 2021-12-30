from packs.postprocessor.timer2 import Timer, organize_results

var=organize_results()
timer=Timer(var)

# timer.export_preprocessor_results(var)
timer.export_final_time()
timer.export_processor_cumulative()

timer.export_table()
