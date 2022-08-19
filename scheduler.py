from multiprocessing.sharedctypes import Value
import os
from datetime import datetime

import xlwings as xw
import pandas as pd
import numpy as np
import cvxpy as cp
from docopt import docopt

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Runs optimization model given Excel inputs
def run_model(staff_info, hyper_params, avail, n_in, b_in, r_in):
    # Sets
    num_weeks, num_staff = avail.shape
    people = list(range(0, num_staff))
    weeks = list(range(0, num_weeks))
    res = staff_info[staff_info.resident].index
    path = staff_info[~staff_info.resident].index

    # Paramaters
    n_target = staff_info.n_target.to_numpy()
    b_target = staff_info.b_target.to_numpy()

    n_rp_freq = n_target[res]/len(path)
    n_rp_target = np.tile(n_rp_freq.reshape((-1,1)), len(path))

    b_rp_freq = b_target[res]/len(path)
    b_rp_target = np.tile(b_rp_freq.reshape((-1,1)), len(path))

    # Decision variables
    n = cp.Variable(avail.shape, boolean=True, name='necropsies')
    b = cp.Variable(avail.shape, boolean=True, name='biopsies')
    r_one = cp.Variable(avail.shape, boolean=True, name='rarc_one')
    r_two = cp.Variable(avail.shape, boolean=True, name='rarc_two')

    # Definition variables
    r = r_one + r_two
    duties = n + b
    tasks = n + b + r
    constraints = []

    # Constraint: Duty assignments based on input
    c_n_in = [n[w,p] >= n_in[w,p] for w in weeks for p in people]
    c_b_in = [b[w,p] >= b_in[w,p] for w in weeks for p in people]
    c_r_in = [r[w,p] >= r_in[w,p] for w in weeks for p in people]
    constraints += c_n_in + c_b_in + c_r_in

    # Constraint: Each duty requires one pathologist
    c_one_path_n = [cp.sum(n[w,path]) == 1 for w in weeks]
    c_one_path_b = [cp.sum(b[w,path]) == 1 for w in weeks]
    constraints += c_one_path_n + c_one_path_b

    # Constraint: Each duty can have a max of one resident
    c_one_res_n = [cp.sum(n[w,res]) <= 1 for w in weeks]
    c_one_res_b = [cp.sum(b[w,res]) <= 1 for w in weeks]
    constraints += c_one_res_n + c_one_res_b

    # Constraint: Residents may be require 2 consecutive weeks of RARC
    c_rarc_1 = [r_one[-1,p] == 0 for p in people]
    c_rarc_2 = [r_one[w,p] == r_two[w+1,p] for w in weeks[:-1] for p in people]
    c_rarc_3 = [r_two[0,p] == 0 for p in people]
    c_rarc_req = [cp.sum(r, axis=0) >= 2 * staff_info.rarc]
    constraints += c_rarc_1 + c_rarc_2 + c_rarc_3 + c_rarc_req

    # Constraint: Cannot work both duties or during time off
    c_work_avail = [tasks[w,p] <= avail.iloc[w,p] for w in weeks for p in people]
    constraints += c_work_avail

    # Definitional Constraint: Two consecutive duties
    nn_first = cp.Variable(avail.shape, boolean=True)
    nn_second = cp.Variable(avail.shape, boolean=True)
    bb_first = cp.Variable(avail.shape, boolean=True)
    bb_second = cp.Variable(avail.shape, boolean=True)
    bn_first = cp.Variable(avail.shape, boolean=True)
    bn_second = cp.Variable(avail.shape, boolean=True)
    nb_first = cp.Variable(avail.shape, boolean=True)
    nb_second = cp.Variable(avail.shape, boolean=True)

    c_nn_doubles = [nn_first[w] + nn_second[w] == cp.sum(n[w:w+2], axis=0) for w in weeks]
    c_bb_doubles = [bb_first[w] + bb_second[w] == cp.sum(b[w:w+2], axis=0) for w in weeks]
    c_bn_doubles = [bn_first[w] + bn_second[w] == b[w] + n[w+1] for w in weeks[:-1]]
    c_nb_doubles = [nb_first[w] + nb_second[w] == n[w] + b[w+1] for w in weeks[:-1]]
    constraints += c_nn_doubles + c_bb_doubles + c_bn_doubles + c_nb_doubles

    # Definitional Constraint: Three consecutive duties
    triple_one_two = cp.Variable(avail.shape, integer=True)
    triple_third = cp.Variable(avail.shape, boolean=True)
    c_triples = [triple_one_two[w] + triple_third[w] == cp.sum(duties[w:w+3], axis=0) for w in weeks]
    constraints += c_triples + [triple_one_two <= 2]

    # Definitional Constraint: Absolute difference between target Bx and scheduled Bx
    b_diff = cp.Variable(num_staff)
    b_below = cp.Variable(num_staff, nonneg=True)
    b_above = cp.Variable(num_staff, nonneg=True)
    c_b_diff = [b_diff == cp.sum(b, axis=0) - b_target]
    c_b_below = [b_below >= cp.neg(b_diff)]
    c_b_above = [b_above >= b_diff]
    constraints += c_b_diff + c_b_below + c_b_above

    # Definitional Constraint: Absolute difference between target Nx and scheduled Nx
    n_diff = cp.Variable(num_staff)
    n_below = cp.Variable(num_staff, nonneg=True)
    n_above = cp.Variable(num_staff, nonneg=True)
    c_n_diff = [n_diff == cp.sum(n, axis=0) - n_target]
    c_n_below = [n_below >= cp.neg(n_diff)]
    c_n_above = [n_above >= n_diff]
    constraints += c_n_diff + c_n_below + c_n_above

    # Definitional Constraint: Shortage of target Nx R/P pairs
    n_pairs = [cp.Variable((len(res), len(path)), boolean=True) for _ in weeks]
    n_pairs_below = cp.Variable((len(res), len(path)), nonneg=True)
    c1_n_pairs = [n_pairs[w][ir,ip] <= n[w,p] for ip,p in enumerate(path) for ir,_ in enumerate(res) for w in weeks]
    c2_n_pairs =[n_pairs[w][ir,ip] <= n[w,r] for ip,_ in enumerate(path) for ir,r in enumerate(res) for w in weeks]
    c3_n_pairs = [n_pairs_below >= n_rp_target - cp.sum(n_pairs, axis=2)]
    constraints += c1_n_pairs + c2_n_pairs + c3_n_pairs

    # Definitional Constraint: Shortage of target Bx R/P pairs
    b_pairs = [cp.Variable((len(res), len(path)), boolean=True) for _ in weeks]
    b_pairs_below = cp.Variable((len(res), len(path)), nonneg=True)
    c1_b_pairs = [b_pairs[w][ir,ip] <= b[w,p] for ip,p in enumerate(path) for ir,_ in enumerate(res) for w in weeks]
    c2_b_pairs =[b_pairs[w][ir,ip] <= b[w,r] for ip,_ in enumerate(path) for ir,r in enumerate(res) for w in weeks]
    c3_b_pairs = [b_pairs_below >= b_rp_target - cp.sum(b_pairs, axis=2)]
    constraints += c1_b_pairs + c2_b_pairs + c3_b_pairs

    # Costs
    double_cost = 4 * cp.sum(nn_second) + 3 * cp.sum(bn_second) + 2 * cp.sum(bb_second) + 1 * cp.sum(nb_second)
    triple_cost = cp.sum(triple_third)
    duty_cost = cp.sum(b_below + b_above + n_below + n_above)
    pair_cost = cp.sum(n_pairs_below + b_pairs_below)

    # Solve
    obj = cp.Minimize(
        hyper_params[0] * double_cost + 
        hyper_params[1] * triple_cost + 
        hyper_params[2] * duty_cost + 
        hyper_params[3] * pair_cost
    )
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='GUROBI', TimeLimit=120)
    return (n, b, r, prob.status)

# Enumeration of status colors
class Color():
    GREEN = (204, 255, 153)
    YELLOW = (255, 255, 153)
    RED = (255, 153, 153)


class Scheduler:
    def __init__(self, workbook=None):
        if isinstance(workbook, xw.Book):
            self.workbook = workbook
        else:
            self.workbook = xw.Book.caller()
        self.input_sheet = self.workbook.sheets['User Input']
        self.output_sheet = self.workbook.sheets['FinalSchedule']
        self.status_cell = self.output_sheet.range('C4')
        self.num_staff = int(self.input_sheet.range('B12').value)
        self.num_weeks = int(self.input_sheet.range('E3').value)
        self.schedule_range = self.output_sheet.range((6, 3), (5 + self.num_weeks, 2 + self.num_staff))
        self.dates_range = self.output_sheet.range((6, 1), (5 + self.num_weeks, 2))

    def get_staff_info(self):
        ### Parse staff info
        self.status_cell.color = Color.YELLOW
        self.status_cell.value = 'Importing Data...'
        num_staff = int(self.input_sheet.range('B12').value)
        staff_info_range = self.input_sheet.range((15, 1), (14 + num_staff, 5))
        staff_info = pd.DataFrame(staff_info_range.value, columns=[
            'full_name',
            'person',
            'target_weeks',
            'position',
            'rarc'
        ])
        # Check to see if we are missing staff values
        missing_staff_values = staff_info.isna().sum().sum() > 0
        if missing_staff_values:
            self.status_cell.color = Color.RED
            self.status_cell.value = 'Missing Data'
            self.workbook.macro('MissingStaffData')()
            raise ValueError
        staff_info['b_target'] = staff_info.target_weeks.astype(int) / 2
        staff_info['n_target'] = staff_info.target_weeks.astype(int) / 2
        staff_info['resident'] = staff_info.position.str.contains('Resident')
        staff_info['rarc'] = staff_info.rarc.str.contains('Yes').astype(int)
        return staff_info

    ### Parse hyperparameters
    def get_hyper_params(self):
        slider_cells = [(5, 8), (9, 8), (13, 8), (17, 8)]
        hyper_params = [self.input_sheet.range(cell).value for cell in slider_cells]
        return hyper_params

    ### Parse schedule inputs
    def get_schedule_data_in(self):
        staff_info = self.get_staff_info()
        headers = staff_info.person
        schedule_data_in = pd.DataFrame(self.schedule_range.value, columns=headers)
        return schedule_data_in

    def generate_schedule(self):
        try:
            staff_info = self.get_staff_info()
        except Exception as _:
            return
        hyper_params = self.get_hyper_params()
        schedule_data_in = self.get_schedule_data_in()

        n_in = (schedule_data_in == 'Necropsy').astype(int).to_numpy()
        b_in = (schedule_data_in == 'Biopsy').astype(int).to_numpy()
        r_in = (schedule_data_in == 'RARC').astype(int).to_numpy()
        avail = (~schedule_data_in.isin(['Time Off', 'Teaching', 'Other'])).astype(int)

        ### Run optimization model
        self.status_cell.value = 'Running model...'
        try:
            n, b, r, prob_status = run_model(staff_info, hyper_params, avail, n_in, b_in, r_in)
        except Exception as _:
            self.status_cell.color = Color.RED
            self.status_cell.value = 'Model Error'
            return

        if prob_status == 'infeasible_or_unbounded':
            self.status_cell.color = Color.RED
            self.status_cell.value = 'Infeasible'
            self.workbook.macro('InfeasibleSol')()
            return
        else:
            self.status_cell.color = Color.GREEN
            self.status_cell.value = 'Schedule Found!'

        ### Populate spreadsheet
        schedule_data_out = schedule_data_in.copy()
        schedule_data_out[n.value.astype(bool)] = 'Necropsy'
        schedule_data_out[b.value.astype(bool)] = 'Biopsy'
        schedule_data_out[r.value.astype(bool)] = 'RARC'

        self.schedule_range.value = schedule_data_out.to_numpy()
        self.workbook.macro('MultipleConditionalFormattingExample')()

    def get_duty_schedule(self):
        def join_cols(row):
            values = (row * ~row.isna()).to_list()
            return ',   '.join(list(filter(lambda x: x != "", values)))

        schedule_data_in = self.get_schedule_data_in()
        schedule_data_in.columns = self.get_staff_info().full_name

        names = schedule_data_in.copy()
        names.iloc[:,:] = schedule_data_in.columns

        necropsies = (names * (schedule_data_in == 'Necropsy')).apply(join_cols, axis=1)
        biopsies = (names * (schedule_data_in == 'Biopsy')).apply(join_cols, axis=1)
        weeks = pd.DataFrame(self.dates_range.value)
        output = pd.concat([weeks, biopsies, necropsies], axis=1)
        output.columns = ['From', 'To', 'Biopsies', 'Necropsies']

        return output


    def generate_pdf(self, cfilepath):
        data = self.get_duty_schedule()
        data['From'] = data['From'].dt.strftime('%m/%d/%y')
        data['To'] = data['To'].dt.strftime('%m/%d/%y')
        schedule_data = data.to_numpy().tolist()
        schedule_data.insert(0, ['From', 'To', 'Biopsy', 'Necropsy'])

        def addPageNumberCreated(canvas, doc):
            addPageNumber(canvas, doc)
            date_created = f"Created: {datetime.now().strftime('%m/%d/%y')}"
            canvas.drawString(0.75 * inch, 0.5 * inch, date_created)

        def addPageNumber(canvas, _):
            page_num = str(canvas.getPageNumber())
            canvas.drawRightString(10.25 * inch, 0.5 * inch, page_num)

        TABLE_STYLE = TableStyle(
            [
                ('BACKGROUND', (0,0), (-1,0), colors.lightcoral),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]
        )

        TITLE_STYLE = ParagraphStyle(
            'Title',
            fontSize=20,
            leading=26,
            alignment=1
        )

        SUBTITLE_STYLE = ParagraphStyle(
            'Subtitle',
            fontSize=14,
            leading=24,
            alignment=1
        )

        t = Table(schedule_data, colWidths=[
            0.7 * inch,
            0.7 * inch,
            4.05 * inch,
            4.05 * inch
        ])
        t.setStyle(TABLE_STYLE)

        logo_path = os.path.join(cfilepath, 'logo.png')
        logo = Image(logo_path, height=90, width=270, hAlign='LEFT')
        title = Paragraph('Anatomic Pathology Duty Schedule', TITLE_STYLE)
        subtitle_text = f"{data.iloc[0,0]} - {data.iloc[-1,1]}"
        subtitle = Paragraph(subtitle_text, SUBTITLE_STYLE)

        story = []
        story.append(logo)
        story.append(title)
        story.append(subtitle)
        story.append(t)

        save_name = f"DutySchedule_{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.pdf"
        save_path = os.path.join(cfilepath, save_name)

        doc = SimpleDocTemplate(
            save_path,
            pagesize=landscape(letter),
            topMargin=0.3*inch,
            bottomMargin=0.6*inch,
            leftMargin=0.35*inch,
            rightMargin=0.35*inch,
        )

        doc.build(
            story,
            onFirstPage=addPageNumberCreated,
            onLaterPages=addPageNumber
        )

        self.status_cell.color = Color.GREEN
        self.status_cell.value = 'Exported!'


def main():
    """
    Usage:
    scheduler.py generate <excelpath>
    scheduler.py pdf <excelpath>
    """
    return 0

if __name__ == '__main__':
    arguments = docopt(main.__doc__)
    print(arguments)
    # if os.path.exists('ScheduleTemplate.xlsm')
    # scheduler = Scheduler()