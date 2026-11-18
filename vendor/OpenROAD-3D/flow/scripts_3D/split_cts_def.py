import os


try:
    results_dir = os.environ['RESULTS_DIR']
    print(f"Results directory is located at: {results_dir}")
except KeyError:
    print("ERROR: RESULTS_DIR environment variable is not set.")


gp_out_file = f"{results_dir}/4_1_cts.def"

upper_path = f"{results_dir}/upper.def"
bottom_path = f"{results_dir}/bottom.def"
with open(gp_out_file, 'r', encoding='utf-8') as def_file, \
        open(upper_path, 'w', encoding='utf-8') as upper_def, \
        open(bottom_path, 'w', encoding='utf-8') as bottom_def:
        part = False
        indicator = False
        num_bot = 0
        num_upper = 0
        net_part = False
        for line in def_file:
            if 'COMPONENTS' in line:
                part = True
            if 'END' in line:
                part = False
            if 'NETS' in line:
                net_part = True

            
            if part:
                if ('PLACED' not in line) or ('FIXED' not in line):
                    class_name = line.split()[2]
                    # print(class_name)
                    if 'bottom' in class_name:
                        indicator = 'bottom'
                        num_bot += 1
                    elif 'upper' in class_name:
                        indicator = 'upper'
                        num_upper += 1
                    
            else:
                if net_part:
                    indicator = None
                else:
                    indicator = 'all'
            if indicator == 'bottom':
                bottom_def.write(line)
            if indicator == 'upper':
                upper_def.write(line)
            if indicator == 'all':
                bottom_def.write(line)
                upper_def.write(line)
            if 'END NETS' in line:
                net_part = False
