#!/bin/python

import sys, json
from collections import OrderedDict


def convert_md_to_tex(md):
    # creating ordered report from dict report
    report = OrderedDict(md['report'])
    ncols = 5
    support_total = report['macro avg']['support']

    # constants
    col_senl = " & "
    row_senl = "\\\\\n"
    hline = "\\hline\n"
    class_label = "Class"
    row_start = '\t\t'

    # latex table header
    table = "\\begin{table}[h!]\n"
    table += "\t\\caption{Class Accuracy: \\textbf {" + md['fid'] + "} (Model: " + md['id'] + ")}\n"
    table += "\t\\label{tab:class_report_" + md['fid'] + "}\n"
    table += "\t\\small\n"
    table += "\t\\begin{tabular}{" + ('c' * ncols) + "}\n"

    theader = row_start + hline + "\\textbf{" + class_label + "}" + col_senl
    tbody = hline
    tfooter = hline + hline

    r = 0
    for rkey, row in report.items():
        # add horizontal rule to separate out averages
        # handle accuracy calc.
        if rkey == 'macro avg':
            rkey = 'cAvg'
            tbody += row_start + hline
        elif rkey == 'weighted avg':
            rkey = 'wAvg'
        if rkey == 'accuracy':
            tfooter += row_start + "\\multicolumn{2}{l}{\\textbf{Pixel Accuracy:} }" + col_senl + "{0:0.3f}".format(
                float(row)) + 2 * col_senl + row_senl
            continue
        # all other metrics in report
        else:
            c = 0
            for ckey, col in row.items():
                # add row label
                if c == 0:
                    tbody += row_start + "\\textbf{" + rkey + "}" + col_senl
                # get column headings from first row
                if r == 0:
                    theader += "\\textbf{" + ckey.capitalize() + "}"
                    if c < ncols - 2:
                        theader += col_senl

                # add column value
                if ckey == 'support':
                    tbody += "{0:0.3f}".format(float(col)/float(support_total))
                else:
                    tbody += "{0:0.3f}".format(float(col))
                # close column
                if c < ncols - 2:
                    tbody += col_senl
                c += 1
        # close header row
        if r == 0:
            theader += row_senl
        # close body row
        tbody += row_senl
        r += 1

    # add other metrics
    tfooter += row_start + "\\multicolumn{2}{l}{\\textbf{F1 Score: }}" + col_senl + "{0:0.3f}".format(float(md['f1'])) + 2 * col_senl + row_senl
    tfooter += row_start + "\\multicolumn{2}{l}{\\textbf{wIoU: }}" + col_senl + "{0:0.3f}".format(float(md['iou'])) + 2 * col_senl + row_senl
    tfooter += row_start + "\\multicolumn{2}{l}{\\textbf{MCC: }}" + col_senl + "{0:0.3f}".format(float(md['mcc'])) + 2 * col_senl + row_senl
    tfooter += row_start + "\\multicolumn{2}{l}{\\textbf{Total Pixels: }}" + col_senl + "{0:0.3f}".format(
        int(support_total)) + 2 * col_senl + row_senl

    # include header and body in table
    table += theader + tbody + tfooter + row_start + hline


    table += "\t\\end{tabular}\n"
    table += "\\end{table}\n"

    return table


