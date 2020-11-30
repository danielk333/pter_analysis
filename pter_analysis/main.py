
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


cfg_help = f'Path to config file, defaults to "{str(CONFIGFILE)}".'
todo_help = 'Path to the todotxt file to analyse, if not given, use the one in the configuration file.'
default_est_help = 'If no "estimate" tag is given, assume this many hours (else exclude those tasks).'


def unescape(x):
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
@click.version_option(version='1.0.1')
def cli():
    pass



@cli.group()
def distribution():
    '''A subset of commands that creates distributions of tasks.
    '''
    pass


@distribution.command()
@click.argument('SEARCH', type=str)
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--default-estimate', default=0, type=float, help=default_est_help)
@click.option('--show-all', is_flag=True, help='Show all projects')
def projects(search, config, default_estimate, show_all, todotxt):
    '''Plots the distribution of spent time, estimated time and task frequency over projects.

    SEARCH: A single pter-type search string.
    '''


    cfg, todo = prepare(todotxt, config)

    tasks = apply_serach(cfg, todo, search)

    dy = 0.1
    rot = 70

    spent, est, num = analysis.distribute_projects(tasks, default_estimate=default_estimate, filter_zero=not show_all)

    fig, ax = plt.subplots()
    ax.set_title('Total time spent per project')
    ax.bar([unescape(x) for x in spent], [spent[x] for x in spent])
    ax.set_ylabel('Time [h]')
    ax.set_xticklabels([unescape(x) for x in spent], rotation=rot)
    pos = ax.get_position()
    pos.y0 += dy
    ax.set_position(pos)

    fig, ax = plt.subplots()
    ax.set_title('Total estimated time left per project')
    ax.bar([unescape(x) for x in est], [est[x] for x in est])
    ax.set_ylabel('Time [h]')
    ax.set_xticklabels([unescape(x) for x in est], rotation=rot)
    pos = ax.get_position()
    pos.y0 += dy
    ax.set_position(pos)

    fig, ax = plt.subplots()
    ax.set_title('Tasks left per project')
    ax.bar([unescape(x) for x in num], [num[x] for x in num])
    ax.set_ylabel('Frequency')
    ax.set_xticklabels([unescape(x) for x in num], rotation=rot)
    pos = ax.get_position()
    pos.y0 += dy
    ax.set_position(pos)

    plt.show()




@distribution.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--default-estimate', default=0, type=float, help=default_est_help)
@click.option('--projects', is_flag=True, help='Divide the distribution into projects')
def ages(config, default_estimate, todotxt, projects, search):
    '''Plots the distribution of time between task creation date and now, i.e task ages.

        [SEARCH]: Pter-type search string(s), multiple searches are given by space separation and surrounded by quotes. The standard GNU syntax of supplying -- before these searches is used.
    '''

    cfg, todo = prepare(todotxt, config)

    for sch in search:
        tasks = apply_serach(cfg, todo, sch)

        if projects:
            task_distribution = analysis.group_projects(tasks)
            ages = []
            for proj in task_distribution:
                ages_ = []
                for task in task_distribution[proj]:
                    if task.creation_date is None:
                        continue
                    dt = datetime.date.today() - task.creation_date
                    ages_.append(dt.days)
                ages.append(ages_)

            fig, ax = plt.subplots()
            ax.set_title(f'{sch} | Task age distribution')
            ax.boxplot(ages)

            xtics = list(task_distribution.keys())
            xtics = [unescape(x) for x in xtics]
            ax.set_xticklabels(xtics, rotation=45)
            
            ax.set_ylabel('Age [d]')
            
            pos = ax.get_position()
            pos.y0 += 0.1
            ax.set_position(pos)

        else:

            ages = []
            for task in tasks:
                if task.creation_date is None:
                    continue
                dt = datetime.date.today() - task.creation_date
                ages.append(dt.days)

            fig, ax = plt.subplots()
            ax.set_title(f'{sch} | Task age distribution')
            ax.hist(ages)
            ax.set_xlabel('Age [d]')
            ax.set_ylabel('Frequency')

    plt.show()


@distribution.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--projects', is_flag=True, help='Divide the distribution into projects')
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
def completion(config, todotxt, projects, search):
    '''Plots the distribution of time between task creation and task completion.

        [SEARCH]: Pter-type search string(s), multiple searches are given by space separation and surrounded by quotes. The standard GNU syntax of supplying -- before these searches is used.
    '''

    cfg, todo = prepare(todotxt, config)

    for sch in search:
        tasks = apply_serach(cfg, todo, sch)

        if projects:
            task_distribution = analysis.group_projects(tasks)
            proj_compl = []
            for proj in task_distribution:
                proj_compl.append(analysis.get_completion_time(task_distribution[proj]))

            fig, ax = plt.subplots()
            ax.set_title(f'{sch} | Task completion time distribution')
            ax.boxplot(proj_compl)

            xtics = list(task_distribution.keys())
            xtics = [unescape(x) for x in xtics]
            ax.set_xticklabels(xtics, rotation=45)
            
            ax.set_ylabel('Completion time [d]')
            
            pos = ax.get_position()
            pos.y0 += 0.1
            ax.set_position(pos)

        else:
            compl = analysis.get_completion_time(tasks)
            fig, ax = plt.subplots()
            ax.set_title(f'{sch} | Task completion time distribution')
            ax.hist(compl)
            ax.set_xlabel('Completion time [d]')
            ax.set_ylabel('Frequency')

    plt.show()



@distribution.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--projects', is_flag=True, help='Divide the distribution into projects')
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--show-all', is_flag=True, help='Show all projects')
def target(config, todotxt, projects, show_all, search):
    '''Plots the distribution of task completion date relative target due date.

        [SEARCH]: Pter-type search string(s), multiple searches are given by space separation and surrounded by quotes. The standard GNU syntax of supplying -- before these searches is used.
    '''

    cfg, todo = prepare(todotxt, config)

    fig, ax = plt.subplots()

    ax = plot_target(ax, cfg, todo, projects, show_all, search)

    plt.show()


def plot_target(ax, cfg, todo, projects, show_all, search, title_add_search=True):

    if cfg.getboolean('General', 'usetex'):
        larr = '$\\leftarrow$'
        rarr = '$\\rightarrow$'
    else:
        larr = '<-'
        rarr = '->'


    for sch in search:
        tasks = apply_serach(cfg, todo, sch)

        if projects:
            task_distribution = analysis.group_projects(tasks)

            proj_compl = []
            for proj in task_distribution:
                proj_compl.append(analysis.get_target(task_distribution[proj]))

            if not show_all:
                o_keys = list(task_distribution.keys())
                task_distribution = {key:task_distribution[key] for i, key in enumerate(o_keys) if len(proj_compl[i]) > 0}
                proj_compl = [proj_compl[i] for i in range(len(o_keys)) if len(proj_compl[i]) > 0]

            
            if title_add_search:
                ax.set_title(f'{sch} | Task completion date relative target date')
            else:
                ax.set_title('Relative task completion date')
            ax.boxplot(proj_compl)

            xtics = list(task_distribution.keys())
            xtics = [unescape(x) for x in xtics]
            ax.set_xticklabels(xtics, rotation=45)
            
            ax.set_ylabel(f'After due {larr} Completed [d] {rarr} Before due')
            
            pos = ax.get_position()
            pos.y0 += 0.1
            ax.set_position(pos)

        else:
            compl = analysis.get_target(tasks)
            if title_add_search:
                ax.set_title(f'{sch} | Task completion date relative target date')
            else:
                ax.set_title('Relative task completion date')
            ax.hist(compl)
            ax.set_xlabel(f'After due {larr} Completed [d] {rarr} Before due')
            ax.set_ylabel('Frequency')

    return ax



@distribution.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--projects', is_flag=True, help='Divide the distribution into projects')
@click.option('--show-all', is_flag=True, help='Show all projects')
def delay(config, projects, show_all, todotxt, search):
    '''Plots the distribution of the "t:" tag relative the "due:" tag, i.e. task delays.

        [SEARCH]: Pter-type search string(s), multiple searches are given by space separation and surrounded by quotes. The standard GNU syntax of supplying -- before these searches is used.
    '''

    cfg, todo = prepare(todotxt, config)

    fig, ax = plt.subplots()

    ax = plot_delays(ax, cfg, projects, show_all, todo, search)

    plt.show()


def plot_delays(ax, cfg, projects, show_all, todo, search, title_add_search=True):

    for sch in search:
        tasks = apply_serach(cfg, todo, sch)

        if projects:
            task_distribution = analysis.group_projects(tasks)
            proj_compl = []
            for proj in task_distribution:
                proj_compl.append(analysis.get_delays(task_distribution[proj]))

            xtics = list(task_distribution.keys())
            xtics = [unescape(x) for x in xtics]

            if show_all:
                xtics = [x for ind, x in enumerate(xtics)]
                proj_compl = [x for x in proj_compl]
            else:
                xtics = [x for ind, x in enumerate(xtics) if len(proj_compl[ind]) > 0]
                proj_compl = [x for x in proj_compl if len(x) > 0]

            if title_add_search:
                ax.set_title(f'{sch} | Task delay time distribution')
            else:
                ax.set_title('Task delay time')
            ax.boxplot(proj_compl)

            ax.set_xticklabels(xtics, rotation=45)
            
            ax.set_ylabel('Delay time [d]')
            
            pos = ax.get_position()
            pos.y0 += 0.1
            ax.set_position(pos)

        else:
            delays = analysis.get_delays(tasks)
            if title_add_search:
                ax.set_title(f'{sch} | Task delay time distribution')
            else:
                ax.set_title('Task delay time')
            ax.hist(delays)
            ax.set_xlabel('Delay time [d]')
            ax.set_ylabel('Frequency')

    return ax

@cli.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--default-estimate', default=0, type=float, help=default_est_help)
def accuracy(config, default_estimate, todotxt, search):
    '''Checks the prediction accuracy for the input search(es).

    [SEARCH]: Pter-type search string(s), multiple searches are given by space separation and surrounded by quotes. The standard GNU syntax of supplying -- before these searches is used.
    '''

    cfg, todo = prepare(todotxt, config)

    if cfg.getboolean('General', 'usetex'):
        larr = '$\\leftarrow$'
        rarr = '$\\rightarrow$'
    else:
        larr = '<-'
        rarr = '->'

    errors = []
    for sch in search:
        tasks = apply_serach(cfg, todo, sch)
        error = analysis.calculate_error(tasks, default_estimate=default_estimate)
        errors += [error]

    fig, ax = plt.subplots()
    ax.set_title('Execution time estimate errors')
    ax.boxplot(errors)
    ax.set_xticklabels(search, rotation=45)
    ax.set_ylabel(f'Overestimate {larr} [h] {rarr} Underestimate')
    
    pos = ax.get_position()
    pos.y0 += 0.24
    ax.set_position(pos)

    plt.show()




@cli.command()
@click.argument('SEARCH', default='')
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--default-estimate', default=0, type=float, help=default_est_help)
@click.option('--show-all', is_flag=True, help='Show all projects')
def next_week(search, config, default_estimate, todotxt, show_all):
    '''Shows the planned work that needs to be completed in the next 7 days distributed across projects, takes "spent" tag into account for non-completed tasks.

    [SEARCH]: A single pter-type search string, surrounded by quotes. If no SEARCH given, uses 't:+1w duebefore:+1w -@delegate -@milestone'.
    '''

    if len(search) == 0:
        search = 't:+1w duebefore:+1w -@delegate -@milestone'

    cfg, todo = prepare(todotxt, config)

    nominal_day = parse_duration(cfg.get('General','work-day-length'))
    nominal_week = parse_duration(cfg.get('General','work-week-length'))

    w_h = nominal_week.days*(nominal_day.total_seconds()/3600.0)
    w_h_ok = w_h*0.75

    tasks = apply_serach(cfg, todo, search)

    dy = 0.1
    rot = 70

    spent, est, num = analysis.distribute_projects(tasks, default_estimate=default_estimate, filter_zero=not show_all)

    fig, ax = plt.subplots()
    ax.set_title('Total estimated work-time next week')
    ax.bar([unescape(x) for x in est] + ['Total'], [est[x] for x in est] + [np.sum(est[x] for x in est)] )
    ax.set_ylabel('Time [h]')
    ax.set_xticklabels([unescape(x) for x in est] + ['[Total]'], rotation=rot)
    pos = ax.get_position()
    pos.y0 += dy
    ax.set_position(pos)

    ax.axhline(y=w_h, color='r')
    ax.axhline(y=w_h_ok, color='g')

    plt.show()




@cli.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--start', default='', type=str, help='Starting date in ISO format')
@click.option('--end', default='', type=str, help='Ending date in ISO format')
@click.option('--out', default='rst', type=str, help='Output format, can be [rst|html|txt]')
def done(config, start, end, out, todotxt, search):
    '''Prints completion lists.

    [SEARCH]: Pter-type search string(s), multiple searches are given by space separation and surrounded by quotes. The standard GNU syntax of supplying -- before these searches is used.
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
            return '- ' + text + '\n'

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

    elif out == 'txt':

        def add_row(text):
            return text + '\n'

        def add_header(text, first):
            out_text = ''
            if not first:
                out_text += '\n'
            out_text += str(text) + '\n'
            out_text += '\n'
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
    elif out == 'txt':
        pass


    print(out_text)



@cli.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--default-estimate', default=0, type=float, help=default_est_help)
@click.option('--end', default='', type=str, help='End date for burndown chart (ISO format)')
@click.option('--default-delay', default=5, type=int, help='If "due" date and "t" date has already passed, assume this many days automatic delay')
@click.option('--adaptive', is_flag=True, help='Use adaptive work distribution algorithm trying to keep 100% activity')
def burndown(config, default_estimate, default_delay, todotxt, end, search, adaptive):
    '''Creates a burn-down chart for the selection.

    [SEARCH]: Pter-type search string(s), multiple searches are given by space separation and surrounded by quotes. The standard GNU syntax of supplying -- before these searches is used.
    '''

    cfg, todo = prepare(todotxt, config)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 4))

    ax = burndown_plot(ax, cfg, default_estimate, default_delay, todo, end, search, adaptive)

    plt.show()


def burndown_plot(ax, cfg, default_estimate, default_delay, todo, end, search, adaptive):

    if len(end) > 0:
        end_date = datetime.date.fromisoformat(end)
    else:
        end_date = None

    
    nominal_day = parse_duration(cfg.get('General','work-day-length'))
    nominal_week = parse_duration(cfg.get('General','work-week-length'))

    h_per_day = nominal_week.days*(nominal_day.total_seconds()/3600.0)/5.0
    
    locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
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
        adaptive = adaptive,
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

    return ax



@cli.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--default-estimate', default=0, type=float, help=default_est_help)
@click.option('--end', default='', type=str, help='End date for burndown chart (ISO format)')
@click.option('--default-delay', default=5, type=int, help='If "due" date and "t" date has already passed, assume this many days automatic delay')
@click.option('--adaptive', is_flag=True, help='Use adaptive work distribution algorithm trying to keep 100% activity')
def quicklook(config, default_estimate, default_delay, todotxt, end, search, adaptive):
    '''Quicklook panel combining burndown, project time left distribution, delay distributions and target completion date distributions.

        [SEARCH]: Pter-type search string(s), multiple searches are given by space separation and surrounded by quotes. The standard GNU syntax of supplying -- before these searches is used.
    '''

    cfg, todo = prepare(todotxt, config)

    fig = plt.figure(figsize=(18, 7))

    ax = fig.add_subplot(2, 1, 1)
    ax = burndown_plot(ax, cfg, default_estimate, default_delay, todo, end, search[1:], adaptive)

    show_all = False
    projects = True

    ax2 = fig.add_subplot(2, 3, 4)
    ax2 = plot_project_time_left(ax2, search[0], cfg, default_estimate, show_all, todo)


    ax3 = fig.add_subplot(2, 3, 5)
    ax3 = plot_delays(ax3, cfg, projects, show_all, todo, [search[0]], title_add_search = False)


    ax4 = fig.add_subplot(2, 3, 6)
    ax4 = plot_target(ax4, cfg, todo, projects, show_all, [search[0]], title_add_search = False)

    plt.subplots_adjust(bottom=0.1, hspace=0.2, top=0.95)

    plt.show()


def plot_project_time_left(ax, search, cfg, default_estimate, show_all, todo):

    tasks = apply_serach(cfg, todo, search)

    dy = 0.1
    rot = 70

    spent, est, num = analysis.distribute_projects(tasks, default_estimate=default_estimate, filter_zero=not show_all)

    ax.set_title('Total estimated time left per project')
    ax.bar([unescape(x) for x in est], [est[x] for x in est])
    ax.set_ylabel('Time [h]')
    ax.set_xticklabels([unescape(x) for x in est], rotation=rot)
    pos = ax.get_position()
    pos.y0 += dy
    ax.set_position(pos)

    return ax


@cli.command()
@click.argument('SEARCH', type=str)
@click.option('--config', default=CONFIGFILE, help=cfg_help)
@click.option('--todotxt', default='', type=str, help=todo_help)
@click.option('--task-limit', default=1, type=int, help='Max number of tasks per unique date')
@click.option('--name-limit', default=20, type=int, help='Max number of characters per task')
@click.option('--context/--no-context', default=False, help='Remove project tags from task')
@click.option('--project/--no-project', default=True, help='Remove context tags from task')
def timeline(config, project, context, name_limit, task_limit, todotxt, search):
    '''Creates a time-line chart for the due dates of the selection (BETA FEATURE).

    SEARCH: A single pter-type search string.
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