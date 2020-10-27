
#standard python
import sys
import datetime
import calendar

#third party imports
import click
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#Pter and pytodotxt
from pytodotxt import TodoTxt, Task
from pter.utils import parse_duration
from pter.searcher import Searcher

#local import
from .config import get_config, CONFIGFILE
from . import analysis

def escape(x):
    x = x.replace('_',' ')
    return x

def prepare(todotxt, config):
    cfg = get_config(config)

    if len(todotxt) == 0:
        todotxt = cfg.get('General', 'todotxt-file')
    if len(todotxt) == 0:
        raise ValueError(f'No todotxt file given nor configured in {config}')

    todo = TodoTxt(todotxt)
    todo.parse()

    if cfg.getboolean('General', 'usetex'):
        plt.rc('text', usetex=True)

    style = cfg.get('General', 'matplotlib-style')
    if len(style) > 0:
        plt.style.use(style)

    return cfg, todo


def apply_serach(cfg, todo, search):
    sch = Searcher(
        search,
        cfg.getboolean('General', 'search-case-sensitive'),
    )

    sel_tasks = [task for task in todo.tasks if sch.match(task)]
    return sel_tasks


@click.group()
@click.version_option()
def cli():
    pass



@cli.group()
def distribution():
    '''A subset of commands that creates distributions of tasks.
    '''
    pass


@distribution.command()
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--default-estimate', default=0, type=float, help='If no "estimate" tag is given, assume this many hours (else exclude those tasks)')
def projects(config, default_estimate, todotxt):
    '''Plots the distribution of spent time, estimated time and task frequency over projects.
    '''


    cfg, todo = prepare(todotxt, config)

    dy = 0.1
    rot = 70

    spent, est, num = analysis.distribute_projects(todo, default_estimate=default_estimate)

    fig, ax = plt.subplots()
    ax.set_title('Total time spent per project')
    ax.bar([escape(x) for x in spent], [spent[x] for x in spent])
    ax.set_ylabel('Time [h]')
    ax.set_xticklabels([escape(x) for x in spent], rotation=rot)
    pos = ax.get_position()
    pos.y0 += dy
    ax.set_position(pos)

    fig, ax = plt.subplots()
    ax.set_title('Total time estimated time left per project')
    ax.bar([escape(x) for x in est], [est[x] for x in est])
    ax.set_ylabel('Time [h]')
    ax.set_xticklabels([escape(x) for x in est], rotation=rot)
    pos = ax.get_position()
    pos.y0 += dy
    ax.set_position(pos)

    fig, ax = plt.subplots()
    ax.set_title('Tasks left per project')
    ax.bar([escape(x) for x in num], [num[x] for x in num])
    ax.set_ylabel('Frequency')
    ax.set_xticklabels([escape(x) for x in num], rotation=rot)
    pos = ax.get_position()
    pos.y0 += dy
    ax.set_position(pos)

    plt.show()



@distribution.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--default-estimate', default=0, type=float, help='If no "estimate" tag is given, assume this many hours (else exclude those tasks)')
def ages(config, default_estimate, todotxt, search):
    '''Plots the distribution of task age.

        SEARCH: Pter-type search strings (multiple searches are given by space separation)
    '''

    cfg, todo = prepare(todotxt, config)

    for sch in search:
        tasks = apply_serach(cfg, todo, sch)

        ages = []
        for task in tasks:
            if task.creation_date is None:
                continue
            dt = datetime.date.today() - task.creation_date
            ages.append(dt.days)

        fig, ax = plt.subplots()
        ax.set_title(f'{sch}: Task age distribution')
        ax.hist(ages)
        ax.set_xlabel('Age [d]')
        ax.set_ylabel('Frequency')

    plt.show()


@distribution.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--projects', is_flag=True, help='Divide the distribution into projects')
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--default-estimate', default=0, type=float, help='If no "estimate" tag is given, assume this many hours (else exclude those tasks)')
def completion(config, default_estimate, todotxt, projects, search):
    '''Plots the distribution of task time to completion.

        SEARCH: Pter-type search strings (multiple searches are given by space separation)
    '''

    cfg, todo = prepare(todotxt, config)

    for sch in search:
        tasks = apply_serach(cfg, todo, sch)

        if projects:
            task_distribution = analysis.group_projects(tasks)
            proj_compl = []
            for proj in task_distribution:
                proj_compl.append(analysis.get_ages(task_distribution[proj]))

            fig, ax = plt.subplots()
            ax.set_title(f'{sch}: Task completion time distribution')
            ax.boxplot(proj_compl)

            xtics = list(task_distribution.keys())
            xtics = [escape(x) for x in xtics]
            ax.set_xticklabels(xtics, rotation=45)
            
            ax.set_ylabel('Completion time [d]')
            
            pos = ax.get_position()
            pos.y0 += 0.1
            ax.set_position(pos)

        else:
            compl = analysis.get_ages(tasks)
            fig, ax = plt.subplots()
            ax.set_title(f'{sch}: Task completion time distribution')
            ax.hist(compl)
            ax.set_xlabel('Completion time [d]')
            ax.set_ylabel('Frequency')

    plt.show()


@distribution.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.option('--projects', is_flag=True, help='Divide the distribution into projects')
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
def delay(config, projects, todotxt, search):
    '''Plots the distribution of task delays.

        SEARCH: Pter-type search strings (multiple searches are given by space separation)
    '''

    cfg, todo = prepare(todotxt, config)

    for sch in search:
        tasks = apply_serach(cfg, todo, sch)

        if projects:
            task_distribution = analysis.group_projects(tasks)
            proj_compl = []
            for proj in task_distribution:
                proj_compl.append(analysis.get_delays(task_distribution[proj]))

            fig, ax = plt.subplots()
            ax.set_title(f'{sch}: Task delay time distribution')
            ax.boxplot(proj_compl)

            xtics = list(task_distribution.keys())
            xtics = [escape(x) for x in xtics]
            ax.set_xticklabels(xtics, rotation=45)
            
            ax.set_ylabel('Delay time [d]')
            
            pos = ax.get_position()
            pos.y0 += 0.1
            ax.set_position(pos)

        else:
            delays = analysis.get_delays(tasks)
            fig, ax = plt.subplots()
            ax.set_title(f'{sch}: Task delay time distribution')
            ax.hist(delays)
            ax.set_xlabel('Delay time [d]')
            ax.set_ylabel('Frequency')

    plt.show()



@cli.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--default-estimate', default=0, type=float, help='If no "estimate" tag is given, assume this many hours (else exclude those tasks)')
def accuracy(config, default_estimate, todotxt, search):
    '''Checks the prediction accuracy for the searches.

    SEARCH: Pter-type search strings (multiple searches are given by space separation)
    '''

    cfg, todo = prepare(todotxt, config)
    errors = []
    for sch in search:
        tasks = apply_serach(cfg, todo, sch)
        error = analysis.calculate_error(tasks, default_estimate=default_estimate)
        errors += [error]

    fig, ax = plt.subplots()
    ax.set_title('Execution time estimate errors')
    ax.boxplot(errors)
    ax.set_xticklabels(search, rotation=45)
    ax.set_ylabel('Overestimate $\\leftarrow$ [h] $\\rightarrow$ Underestimate')
    
    pos = ax.get_position()
    pos.y0 += 0.24
    ax.set_position(pos)

    plt.show()



@cli.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--start', default='', type=str, help='Starting date in ISO format')
@click.option('--end', default='', type=str, help='Ending date in ISO format')
@click.option('--out', default='rst', type=str, help='Output format, can be [rst|html|txt]')
def done(config, start, end, out, todotxt, search):
    '''Prints completion lists.

    SEARCH: Pter-type search strings (multiple searches are given by space separation)
    '''


    cfg, todo = prepare(todotxt, config)
    tasks = []
    for sch in search:
        tasks += apply_serach(cfg, todo, sch)

    tasks = [task for task in tasks if task.completion_date is not None]
    if len(start) > 0:
        start = datetime.date.fromisoformat(start)
        tasks = [
            task for task in tasks 
            if task.completion_date >= start
        ]
    if len(end) > 0:
        end = datetime.date.fromisoformat(end)
        tasks = [
            task for task in tasks 
            if task.completion_date <= end
        ]

    tasks.sort(key = lambda x: x.completion_date)

    out_text = ''
    if out == 'rst':

        def add_row(text):
            return '- ' + text

        def add_header(text, first):
            out_text = ''
            if not first:
                out_text += '\n'*3
            out_text += str(text) + '\n'
            out_text += '='*len(str(text))
            out_text += '\n'*2
            return out_text

    elif out == 'html':

        out_text += '<html><body>\n'

        def add_row(text):
            return '<li>' + text + '</li>\n'

        def add_header(text, first):
            out_text = ''
            if not first:
                out_text += '</ul>\n'
            out_text += '<h2>' + str(text) + '</h2>\n'
            out_text += '<ul>\n'
            return out_text

    else:
        raise NotImplementedError('Sorry not implemented yet')


    cdate = None
    for ti,task in enumerate(tasks):
        if cdate is None:
            cdate = task.completion_date
            new_header = True
        else:
            if cdate != task.completion_date:
                new_header = True
            else:
                new_header = False

        if new_header:
            out_text += add_header(task.completion_date, first = ti==0)

        out_text += add_row(task.description)

        cdate = task.completion_date


    if out == 'rst':
        pass
    elif out == 'html':
        out_text += '</ul></body></html>'


    print(out_text)



@cli.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.option('--end', default='', type=str, help='End date for burndown chart (ISO format)')
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--default-estimate', default=0, type=float, help='If no "estimate" tag is given, assume this many hours (else exclude those tasks)')
@click.option('--default-delay', default=5, type=int, help='If "due" date and "t" tag is already passed, assume this many days delay')
def burndown(config, default_estimate, default_delay, todotxt, end, search):
    '''Creates a burn-down chart for the selection.

    SEARCH: Pter-type search strings (multiple searches are given by space separation)
    '''

    if len(end) > 0:
        end_date = datetime.date.fromisoformat(end)
    else:
        end_date = None

    cfg, todo = prepare(todotxt, config)
    
    nominal_day = parse_duration(cfg.get('General','work-day-length'))
    nominal_week = parse_duration(cfg.get('General','work-week-length'))

    h_per_day = nominal_week.days*(nominal_day.seconds/3600.0)/5.0


    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 4))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    axv_legend = False

    max_activity = 100

    total_activity = None
    total_dates = None

    all_tasks = [apply_serach(cfg, todo, sch) for sch in search]
    dates, all_activity, all_start, all_end, total_activity = analysis.calculate_total_activity(
        all_tasks, 
        h_per_day, 
        default_estimate=default_estimate, 
        end_date=end_date, 
        default_delay=default_delay,
    )
        
    for i, sch in enumerate(search):
        activity, start, end = all_activity[i], all_start[i], all_end[i]

        for ind in range(len(start)):
            if axv_legend:
                ax.axvline(start[ind], color='g', alpha=0.1)
                ax.axvline(end[ind], color='r', alpha=0.1)
            else:
                ax.axvline(start[ind], color='g', alpha=0.1, label='Task start')
                ax.axvline(end[ind], color='r', alpha=0.1, label='Task due')
                axv_legend = True
        

        if total_activity.max()*100 > max_activity:
                max_activity = total_activity.max()*100
        ax.plot(dates, activity*100, label=sch)

    if len(search) > 1:
        ax.plot(dates, total_activity*100, '--k', label='Total activity')

    ax.set_title('Nominal workload task burn-down')
    ax.set_ylabel('Full-time workload [\%]')
    ax.set_ylim([0,max_activity])
    ax.set_xlim([datetime.datetime.today(),end_date])
    ax.legend()

    plt.show()


@cli.command()
@click.argument('SEARCH', type=str)
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--task-limit', default=1, type=int, help='Max number of tasks per unique date')
@click.option('--name-limit', default=20, type=int, help='Max number of characters per task')
@click.option('--context/--no-context', default=False, help='Remove project tags from task')
@click.option('--project/--no-project', default=True, help='Remove context tags from task')
@click.option('--config', default=str(CONFIGFILE), help='Path to config file')
def timeline(config, project, context, name_limit, task_limit, todotxt, search):
    '''Creates a timeline chart for the due dates of the selection.

    SEARCH: Pter-type search string
    '''
    cfg, todo = prepare(todotxt, config)
    tasks = apply_serach(cfg, todo, search)
    
    def rem_tag(description, regex):
        while True:
            matches = parse_tags(description, regex)
            if len(matches) == 0:
                break
            for match in matches:
                start, end = match.span()
                description = description[:start] + description[end:]
                break
        return description

    def parse_tags(description, regex):
        matches = []
        if description is None:
            return matches

        for match in regex.finditer(description):
            if match:
                matches.append(match)
            else:
                break
        return matches 

    dates = []
    names = []
    num = []
    for task in tasks:
        if 'due' not in task.attributes:
            continue
        due = task.attributes['due'][0].strip()
        due = due.replace(',','')
        date = datetime.date.fromisoformat(due)

        name = task.description

        if not project:
            name = rem_tag(name, Task.PROJECT_RE)
        if not context:
            name = rem_tag(name, Task.CONTEXT_RE)
        name_short = name[:name_limit]

        if len(name) > name_limit:
            name_short += '...'

        if date in dates:
            ind = dates.index(date)
            num[ind] += 1
            if num[ind] > task_limit:
                names[ind] = [f'{num[ind]} Tasks']
            else:
                names[ind].append(name_short)
        else:
            dates.append(date)
            num.append(1)
            if task_limit == 0:
                names.append([f'1 Tasks'])
            else:
                names.append([name_short])

    if len(dates) == 0:
        raise ValueError('No tasks found with due-dates in search')

    levs = [-5, 5, -3, 3, -1, 1]
    if len(dates) < len(levs):
        levs = levs[:len(dates)]

    levels = np.tile(
        levs, 
        int(np.ceil(len(dates)/len(levs))),
    )[:len(dates)]

    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title=search)

    markerline, stemline, baseline = ax.stem(dates, levels,
                                             linefmt="C3-", basefmt="k-",
                                             use_line_collection=True)

    plt.setp(markerline, mec="k", mfc="w", zorder=3)

    # Shift the markers to the baseline by replacing the y-data by zeros.
    markerline.set_ydata(np.zeros(len(dates)))
    ax.set_ylim(-7,7)

    # annotate lines
    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    for d, l, r, va in zip(dates, levels, names, vert):
        for ry, name in enumerate(r):
            ax.annotate(name, xy=(d, l-ry), xytext=(-3, np.sign(l)*3),
                        textcoords="offset points", va=va, ha="right")

    # format xaxis with 4 month intervals
    ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y axis and spines
    ax.get_yaxis().set_visible(False)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.margins(y=0.1)
    plt.show()