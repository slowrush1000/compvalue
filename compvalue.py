import sys
import logging
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self):
        self.m_size         = 2
        self.m_values       = [0.0]*self.m_size
        self.m_has_values   = [False]*self.m_size
        self.m_diffs        = [0.0]*self.m_size
        self.m_diff_percentages     = [0.0]*self.m_size
    def __init__(self, size):
        self.m_size         = size
        self.m_values       = [0.0]*self.m_size
        self.m_has_values   = [False]*self.m_size
        self.m_diffs        = [0.0]*self.m_size
        self.m_diff_percentages     = [0.0]*self.m_size
    def GetSize(self):
        return self.m_size
    def SetValue(self, pos, value):
        self.m_values[pos]  = value
        self.m_has_values[pos] = True
    def GetValue(self, pos):
        return self.m_values[pos]
    def HasValue(self, pos):
        return self.m_has_values[pos]
    def HasAllValues(self):
        for pos in range(0, self.m_size):
            if False == self.m_has_values[pos]:
                return False
        return True
    def SetDiff(self, pos, diff):
        self.m_diffs[pos] = diff
    def GetDiff(self, pos):
        return self.m_diffs[pos]
    def SetDiffPercentage(self, pos, diff_percentage):
        self.m_diff_percentages[pos] = diff_percentage
    def GetDiffPercentage(self, pos):
        return self.m_diff_percentages[pos]

class CompareInfo:
    def __init__(self):
        self.m_size         = 0
        self.m_diff_sum     = 0.0
        self.m_diff_max     = -sys.float_info.max
        self.m_diff_max_key = ''
        self.m_diff_min     = sys.float_info.max
        self.m_diff_min_key = ''
        self.m_diff_percentage_max  = -sys.float_info.max
        self.m_diff_percentage_min  = sys.float_info.max
        self.m_diff_percentage_max_key  = ''
        self.m_diff_percentage_min_key  = ''
        self.m_diff_max_values  = [ False, False ]
        self.m_diff_min_values  = [ False, False ]
        self.m_diff_percentage_max_values  = [ False, False ]
        self.m_diff_percentage_min_values  = [ False, False ]
    def GetSize(self):
        return self.m_size
    def GetDiffSum(self):
        return self.m_diff_sum
    def GetDiffMax(self):
        return self.m_diff_max
    def GetDiffMaxKey(self):
        return self.m_diff_max_key
    def GetDiffMin(self):
        return self.m_diff_min
    def GetDiffMinKey(self):
        return self.m_diff_min_key
    def GetDiffPercentageMax(self):
        return self.m_diff_percentage_max
    def GetDiffPercentageMin(self):
        return self.m_diff_percentage_min
    def GetDiffPercentageMaxKey(self):
        return self.m_diff_percentage_max_key
    def GetDiffPercentageMinKey(self):
        return self.m_diff_percentage_min_key
    def GetDiffMaxValues(self):
        return self.m_diff_max_values
    def GetDiffMinValues(self):
        return self.m_diff_min_values
    def GetDiffPercentageMaxValues(self):
        return self.m_diff_percentage_max_values
    def GetDiffPercentageMinValues(self):
        return self.m_diff_percentage_min_values
    def Run(self, name, value_0th, value_nth, pos):
        self.m_size = self.m_size + 1
        diff    = value_nth - value_0th
        self.m_diff_sum     = self.m_diff_sum + diff
        self.GetMax(name, diff, value_0th, value_nth, pos)
        self.GetMin(name, diff, value_0th, value_nth, pos)
        if math.isclose(value_0th, 0.0):
            return [ diff, 0.0 ]
        diff_percentage  = 100.0 * diff / value_0th
        self.GetMaxPercentage(name, diff_percentage, value_0th, value_nth, pos)
        self.GetMinPercentage(name, diff_percentage, value_0th, value_nth, pos)
        return [ diff, diff_percentage ]
    def GetMax(self, name, diff, value_0th, value_nth, pos):
        if diff > self.m_diff_max:
            self.m_diff_max = diff
            self.m_diff_max_key = name
            self.m_diff_max_values[0]   = value_0th
            self.m_diff_max_values[1]   = value_nth
    def GetMin(self, name, diff, value_0th, value_nth, pos):
        if diff < self.m_diff_min:
            self.m_diff_min = diff
            self.m_diff_min_key = name
            self.m_diff_min_values[0]   = value_0th
            self.m_diff_min_values[1]   = value_nth
    def GetMaxPercentage(self, name, diff_percentage, value_0th, value_nth, pos):
        if diff_percentage > self.m_diff_percentage_max:
            self.m_diff_percentage_max = diff_percentage
            self.m_diff_percentage_max_key = name
            self.m_diff_percentage_max_values[0]   = value_0th
            self.m_diff_percentage_max_values[1]   = value_nth
    def GetMinPercentage(self, name, diff_percentage, value_0th, value_nth, pos):
        if diff_percentage < self.m_diff_percentage_min:
            self.m_diff_percentage_min = diff_percentage
            self.m_diff_percentage_min_key = name
            self.m_diff_percentage_min_values[0]   = value_0th
            self.m_diff_percentage_min_values[1]   = value_nth
    def GetResultStr(self):
            str     =  f'    size                 : {self.GetSize()}\n'
            str     += f'    diff avg             : {float(self.GetDiffSum()) / float(self.GetSize()):e}\n'
            str     += f'    diff max             : {float(self.GetDiffMax()):e}\n'
            str     += f'    diff max name        : {self.GetDiffMaxKey()}\n'
            str     += f'    diff max 0th value   : {self.GetDiffMaxValues()[0]:e}\n'
            str     += f'    diff max nth value   : {self.GetDiffMaxValues()[1]:e}\n'
            str     += f'    diff min             : {float(self.GetDiffMin())}\n'
            str     += f'    diff min name        : {self.GetDiffMinKey()}\n'
            str     += f'    diff min 0th value   : {self.GetDiffMinValues()[0]:e}\n'
            str     += f'    diff min nth value   : {self.GetDiffMinValues()[1]:e}\n'
            str     += f'    diff max %           : {float(self.GetDiffPercentageMax()):.1f}\n'
            str     += f'    diff max % name      : {self.GetDiffPercentageMaxKey()}\n'
            str     += f'    diff max % 0th value : {self.GetDiffPercentageMaxValues()[0]:e}\n'
            str     += f'    diff max % nth value : {self.GetDiffPercentageMaxValues()[1]:e}\n'
            str     += f'    diff min %           : {float(self.GetDiffPercentageMin()):.1f}\n'
            str     += f'    diff min % name      : {self.GetDiffPercentageMinKey()}\n'
            str     += f'    diff min % 0th value : {self.GetDiffPercentageMinValues()[0]:e}\n'
            str     += f'    diff min % nth value : {self.GetDiffPercentageMinValues()[1]:e}\n'
            return str

class Compvalue:
    def __init__(self):
        self.m_version          = '20230730.0.0'
        self.m_node_dic         = {}
        self.m_compare_info     = []
        self.m_output_prefix    = ''
        self.m_size             = 2
        self.m_filenames        = ['']
        self.m_name_poss        = [0]
        self.m_value_poss       = [0]
        self.m_abs              = False
        self.m_graph            = False
        self.m_saveimage        = False
        self.m_logger           = 0
    def Run(self, argc, argv):
        if 9 > argc:
            self.PrintUsage()
            return
        self.InitLogging(argc, argv)
        self.m_logger.info('# compvalue.py(%s) start. ... %s',
                    self.m_version,
                    datetime.datetime.now())
        self.ReadArgv(argc, argv)
        self.PrintInput()
        self.ReadFiles()
        self.Compares()
        self.PrintResult()
        self.WriteOutputFile()
        if True == self.m_graph:
            self.DrawScatterPlot()
            self.DrawValueHistogram()
            self.DrawDiffHistogram()
        self.m_logger.info('# compvalue.py(%s) end. ... %s',
                    self.m_version,
                    datetime.datetime.now())
    def PrintUsage(self):
        print(f'compvalue.py usage({self.m_version}):\n')
        print(f'    % python3 compvalue.py output_prefix size', end = '')
        print(f' 1st_file 1st_name_pos 1st_value_pos', end = '')
        print(f' 2nd_file 2nd_name_pos 2nd_value_pos', end = '')
        print(f'<nth_file nth_name_pos nth_value_pos...> <-abs> <-graph> <-saveimage>')
        print(f'        output_prefix   : output prefix')
        print(f"        size            : comparing data set's size")
        print(f'        nth_file        : file')
        print(f'        nth_name_pos    : name column pos(0-)')
        print(f'        nth_value_pos   : value column pos(0-)')
        print(f'        -abs            : change value to absolute value')
        print(f'        -graph          : make scatter/histogram plot')
        print(f'        -saveimage      : save scatter/histogram plot image')
        print(f'    output file')
        print(f'        output_prefix.log               : log file')
        print(f'        output_prefix.both.txt          : compared result file')
        print(f'        output_prefix.value.scatter.png : value scatter plot image file if you use -graph -saveimage')
        print(f'        output_prefix.value.hist.png    : value hist image file if you use -graph -saveimage')
        print(f'        output_prefix.diff.hist.png     : diff hist image file if you use -graph -saveimage')
        print(f'    example')
        print(f'        compare 2 data set')
        print(f'        % python3 compvalue.py compare_2 2 1.txt 0 10 2.txt 1 8')
        print(f'        compare 3 data set')
        print(f'        % python3 compvalue.py compare_3 3 1.txt 0 10 2.txt 1 8 3.txt 5 20')
        print(f'        compare 3 data set and display scatter/histogram chart')
        print(f'        % python3 compvalue.py compare_3 3 1.txt 0 10 2.txt 1 8 3.txt 5 20 -graph')
        print(f'        compare 3 data set and save scatter/histogram chart image')
        print(f'        % python3 compvalue.py compare_3 3 1.txt 0 10 2.txt 1 8 3.txt 5 20 -graph -saveimage')
        print('')
    def InitLogging(self, argc, argv):
        self.m_output_prefix    = argv[1]
        log_filename    = self.m_output_prefix + '.log'
        logging_format  = logging.Formatter('%(message)s')
        #
        self.m_logger   = logging.getLogger(__name__)
        self.m_logger.setLevel(logging.INFO)
        #
        stream_handler  = logging.StreamHandler()
        stream_handler.setFormatter(logging_format)
        #
        file_handler    = logging.FileHandler(log_filename, 'w')
        file_handler.setFormatter(logging_format)
        #
        self.m_logger.addHandler(stream_handler)
        self.m_logger.addHandler(file_handler)
        #
        self.m_logger.info('')
        self.m_logger.info('# init logging ok. ... %s', 
                    datetime.datetime.now())
        self.m_logger.info('')
    def ReadArgv(self, argc, argv):
        self.m_logger.info('# read args. ... %s',
                           datetime.datetime.now())
        self.m_output_prefix    = argv[1]
        self.m_size             = int(argv[2])
        self.m_compare_info     = ['']*self.m_size
        self.m_filenames        = ['']*self.m_size
        self.m_name_poss        = [0]*self.m_size
        self.m_value_poss        = [0]*self.m_size
        for pos in range(3, argc):
            if (0 == int(pos % 3)) & ('-' != argv[pos][0]):
                index = int(pos / 3) - 1
                self.m_filenames[index]     = argv[pos]
                self.m_name_poss[index]     = int(argv[pos + 1])
                self.m_value_poss[index]    = int(argv[pos + 2])
                pos = pos + 3
            elif '-abs' == argv[pos]:
                self.m_abs = True
            elif '-graph' == argv[pos]:
                self.m_graph = True
            elif '-saveimage' == argv[pos]:
                self.m_saveimage = True
        self.m_logger.info('')
    def PrintInput(self):
        self.m_logger.info('# print input. ... %s',
                           datetime.datetime.now())
        self.m_logger.info('    outpuit prefix : %s', self.m_output_prefix)
        self.m_logger.info('    size           : %d', self.m_size)
        for pos in range(0, len(self.m_filenames)):
            self.m_logger.info('    %d th filename  : %s', pos, self.m_filenames[pos])
            self.m_logger.info('    %d th name pos  : %d', pos, self.m_name_poss[pos])
            self.m_logger.info('    %d th value pos : %d', pos, self.m_value_poss[pos])
        self.m_logger.info('    abs            : %s', self.m_abs)
        self.m_logger.info('    graph          : %s', self.m_graph)
        self.m_logger.info('    saveimage      : %s', self.m_saveimage)
        self.m_logger.info('')
    def ReadFiles(self):
        pos = 0
        for filename in self.m_filenames:
            self.ReadFile(filename, pos)
            pos = pos + 1
    def ReadFile(self, filename, pos):
        self.m_logger.info('# read file(%s). ... %s', filename,
                           datetime.datetime.now())
        file = open(filename, 'rt')
        lines   = file.readlines()
        n_lines = 0
        for line in lines:
            n_lines = n_lines + 1
            if 0 == (n_lines % 1000000000):
                logging.info(f'    {n_lines} lines')
            line = line.strip()
            if '*' == line[0]:
                continue
            tokens = line.split()
            if len(tokens) <= max(int(self.m_name_poss[pos]), int(self.m_value_poss[pos])):
                continue
            name    = tokens[self.m_name_poss[pos]]
            value   = float(tokens[self.m_value_poss[pos]])
            if True == self.m_abs:
                value = abs(value)
            if 0 == pos:
                node    = Node(self.m_size)
                node.SetValue(pos, value)
                self.m_node_dic[name]       = node
            else:
                if name in self.m_node_dic:
                    node = self.m_node_dic[name]
                    node.SetValue(pos, value)
        self.m_logger.info(f'    {n_lines} lines')
        file.close()
        self.m_logger.info('')
    def Compares(self):
        for pos in range(1, len(self.m_filenames)):
            self.Compare(pos)
    def Compare(self, pos):
        self.m_logger.info(f'# compare ( %s vs %s ). ... %s', 
                           self.m_filenames[0], 
                           self.m_filenames[pos],
                           datetime.datetime.now())
        compare_info    = CompareInfo()
        for key in self.m_node_dic:
            node        = self.m_node_dic[key]
            if False == node.HasAllValues():
                continue
            [ diff, diff_percetage ] = compare_info.Run(key, node.GetValue(0), node.GetValue(pos), pos)
            node.SetDiff(pos, diff)
            node.SetDiffPercentage(pos, diff_percetage)
        self.m_compare_info[pos]    = compare_info
        self.m_logger.info('')
    def PrintResult(self):
        self.m_logger.info(f'# print compare result. ... %s', datetime.datetime.now())
        for pos in range(1, len(self.m_filenames)):
            self.m_logger.info('%s th compare result', pos)
            self.m_logger.info('%s', self.m_compare_info[pos].GetResultStr())
    def WriteOutputFile(self):
        filename = self.m_output_prefix + '.both.txt'
        self.m_logger.info(f'# write output file(%s). ... %s',
                           filename,
                           datetime.datetime.now())
        file = open(filename, 'wt')
        # write head
        file.write('*name')
        file.write(f' 0_value')
        for pos in range(1, len(self.m_filenames)):
            file.write(f' {pos}_value')
        for pos in range(1, len(self.m_filenames)):
            file.write(f' {pos}_diff')
        for pos in range(1, len(self.m_filenames)):
            file.write(f' {pos}_diff_pecentage')
        file.write('\n')
        # write head
        for key in self.m_node_dic:
            node = self.m_node_dic[key]
            if False == node.HasAllValues():
                continue
            file.write(f'{key}')
            file.write(f' {float(node.GetValue(0))}')
            for pos in range(1, len(self.m_filenames)):
                file.write(f' {float(node.GetValue(pos))}')
            for pos in range(1, len(self.m_filenames)):
                file.write(f' {float(node.GetDiff(pos))}')
            for pos in range(1, len(self.m_filenames)):
                file.write(f' {float(node.GetDiffPercentage(pos))}')
            file.write('\n')
        file.close()
        self.m_logger.info('')
    def DrawScatterPlot(self):
        self.m_logger.info(f'# draw scatter plot. ... {datetime.datetime.now()}')
        #
        data    = []
        for pos in range(0, len(self.m_filenames)):
            data_x  = [0.0]*self.m_size
            data.append(data_x)
        #
        count = 0
        for key in self.m_node_dic:
            node = self.m_node_dic[key]
            if False == node.HasAllValues():
                continue
            for pos in range(0, len(self.m_filenames)):
                data[pos][count]  = node.GetValue(pos)
            count = count + 1
        #
        data_min    = sys.float_info.max
        data_max    = -sys.float_info.max
        for pos in range(0, len(self.m_filenames)):
            min_1   = min(data[pos])
            if data_min < min_1:
                data_min    = min_1
            max_1   = max(data[pos])
            if data_max > max_1:
                data_max    = max_1
        #
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for pos in range(1, len(self.m_filenames)):
            ax1.scatter(data[0], data[pos], label=f'{self.m_filenames[pos]}:{self.m_name_poss[pos]}:{self.m_value_poss[pos]}')
        ax1.set_title('value scatter plot')
        plt.legend(loc='upper left')
        plt.xlabel(f'{self.m_filenames[0]}:{self.m_name_poss[0]}:{self.m_value_poss[0]}')
        if False == self.m_saveimage:
            plt.show()
        else:
            filename = self.m_output_prefix + '.value.scatter.png'
            self.m_logger.info(f'   scatter plot image file : {filename}')
            plt.savefig(filename)
        self.m_logger.info(f'')
    def DrawValueHistogram(self):
        self.m_logger.info(f'# draw value histrogram plot. ... {datetime.datetime.now()}')
        #
        data    = []
        for pos in range(0, len(self.m_filenames)):
            data_x  = [0.0]*self.m_size
            data.append(data_x)
        #
        count = 0
        for key in self.m_node_dic:
            node = self.m_node_dic[key]
            if False == node.HasAllValues():
                continue
            for pos in range(0, len(self.m_filenames)):
                data[pos][count]  = node.GetValue(pos)
            count = count + 1
        #
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        labels  = []
        for pos in range(0, len(self.m_filenames)):
            labels.append(f'{self.m_filenames[pos]}:{self.m_name_poss[pos]}:{self.m_value_poss[pos]}')
        n_bins = 10
        ax1.hist(data, n_bins, label=labels)
        ax1.set_title('value histogram')
        plt.legend(loc='upper right')
        if False == self.m_saveimage:
            plt.show()
        else:
            filename = self.m_output_prefix + '.value.hist.png'
            self.m_logger.info(f'   value histogram image file : {filename}')
            plt.savefig(filename)
        self.m_logger.info(f'')
    def DrawDiffHistogram(self):
        self.m_logger.info(f'# draw diff histrogram plot. ... {datetime.datetime.now()}')
        #
        data    = []
        for pos in range(0, len(self.m_filenames)):
            data_x  = [0.0]*self.m_size
            data.append(data_x)
        #
        count = 0
        for key in self.m_node_dic:
            node = self.m_node_dic[key]
            if False == node.HasAllValues():
                continue
            for pos in range(1, len(self.m_filenames)):
                data[pos][count]  = node.GetDiff(pos)
            count = count + 1
        #
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        labels  = []
        for pos in range(1, len(self.m_filenames)):
            labels.append(f'{self.m_filenames[pos]}:{self.m_name_poss[pos]}:{self.m_value_poss[pos]}')
        n_bins = 10
        ax1.hist(data[1:len(self.m_filenames)], n_bins, label=labels)
        ax1.set_title('diff histogram')
        plt.legend(loc='upper right')
        if False == self.m_saveimage:
            plt.show()
        else:
            filename = self.m_output_prefix + '.diff.hist.png'
            self.m_logger.info(f'   diff histogram image file : {filename}')
            plt.savefig(filename)
        self.m_logger.info(f'')

def main(argc, argv):
    compvalue   = Compvalue()
    compvalue.Run(argc, argv)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
