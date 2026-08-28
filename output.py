##############################################################################################
##############################################################################################
##############################################################################################

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools

import model_training

##############################################################################################
##############################################################################################
##############################################################################################


def get_primary_results():
    '''
    Conduct the primary part of the investigation:
    100 trials of model training for each setting 'all', 'random' and 'craig'
    '''

    # Set up the parameters (primary results)
    dataset = 'mnist'
    batch_size = 32
    subset_size = 0.4
    epochs = 15
    learn_rate = 1e-2
    weight_decay = 1e-4
    runs = 100
    save_folder = 'results'

    model_training.collect_data(dataset=dataset, batch_size=batch_size, subset_size=subset_size,
                                epochs=epochs, learn_rate=learn_rate, weight_decay=weight_decay,
                                setting='craig', to_collect=['test_loss', 'accuracy', 'epoch_duration', 
                                                             'class_accuracies', 'fpr', 'fnr'],
                                runs=runs, save_files=[f'{save_folder}/craig_{i}' for i in range(runs)])

    model_training.collect_data(dataset=dataset, batch_size=batch_size, subset_size=subset_size,
                                epochs=epochs, learn_rate=learn_rate, weight_decay=weight_decay,
                                setting='random', to_collect=['test_loss', 'accuracy', 'epoch_duration', 
                                                              'class_accuracies', 'fpr', 'fnr'],
                                runs=runs, save_files=[f'{save_folder}/random_{i}' for i in range(runs)])

    model_training.collect_data(dataset=dataset, batch_size=batch_size, subset_size=1.0,
                                epochs=epochs, learn_rate=learn_rate, weight_decay=weight_decay,
                                setting='all', to_collect=['test_loss', 'accuracy', 'epoch_duration', 
                                                           'class_accuracies', 'fpr', 'fnr'],
                                runs=runs, save_files=[f'{save_folder}/all_{i}' for i in range(runs)])


##############################################################################################
##############################################################################################
##############################################################################################


def get_extension_results():
    '''
    Conduct the extended part of the investigation:
    A single trials of model training for each setting 'all', 'random' and 'craig',
    using each subset size of interest.
    '''

    # Set up the parameters (primary results)
    dataset = 'mnist'
    batch_size = 32
    subset_sizes = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    epochs = 50
    learn_rate = 1e-2
    weight_decay = 1e-4
    runs_per = 1
    runs = len(subset_sizes) * runs_per
    save_folder = 'extension_results'

    model_training.collect_data(dataset=dataset, batch_size=batch_size, subset_size=subset_sizes*runs_per,
                                epochs=epochs, learn_rate=learn_rate, weight_decay=weight_decay,
                                setting='craig', to_collect=['test_loss', 'accuracy', 'epoch_duration', 
                                                             'class_accuracies', 'fpr', 'fnr'],
                                runs=runs,
                                save_files=[f'{save_folder}/craig_{(subset_sizes*runs_per)[i]}_{i // len(subset_sizes)}' 
                                            for i in range(runs)])

    model_training.collect_data(dataset=dataset, batch_size=batch_size, subset_size=subset_sizes*runs_per,
                                epochs=epochs, learn_rate=learn_rate, weight_decay=weight_decay,
                                setting='random', to_collect=['test_loss', 'accuracy', 'epoch_duration', 
                                                                'class_accuracies', 'fpr', 'fnr'],
                                runs=runs,
                                save_files=[f'{save_folder}/random_{(subset_sizes*runs_per)[i]}_{i // len(subset_sizes)}' 
                                            for i in range(runs)])

    model_training.collect_data(dataset=dataset, batch_size=batch_size, subset_size=1.0,
                                epochs=epochs, learn_rate=learn_rate, weight_decay=weight_decay,
                                setting='all', to_collect=['test_loss', 'accuracy', 'epoch_duration', 
                                                            'class_accuracies', 'fpr', 'fnr'],
                                runs=runs_per,
                                save_files=[f'{save_folder}/all_{i // len(subset_sizes)}' 
                                            for i in range(runs)])


##############################################################################################
##############################################################################################
##############################################################################################

def load_data_primary(runs):
    '''
    Load in the data from the primary part of this investigation.

    Parameters
    - runs: int, the number of runs from the original data that should be loaded in

    Returns
    - all_data_raw: dict, contains all collected data for the 'all' setting
    - random_data_raw: dict, contains all collected data for the 'random' setting
    - craig_data_raw: dict, contains all collected data for the 'craig' setting

    Notes
    - Assumes that the data is coming from the way get_primary_results saves the files
    '''

    all_data_raw = {
        'test_loss': None,
        'accuracy': None,
        'class_accuracies': None,
        'fpr': None,
        'fnr': None,
        'epoch_duration': None
    }

    random_data_raw = {
        'test_loss': None,
        'accuracy': None,
        'class_accuracies': None,
        'fpr': None,
        'fnr': None,
        'epoch_duration': None
    }

    craig_data_raw = {
        'test_loss': None,
        'accuracy': None,
        'class_accuracies': None,
        'fpr': None,
        'fnr': None,
        'epoch_duration': None
    }

    # Cleverly extracts all the data from the 'results' folder and collects them into
    # the above dictionaries
    for setting, i in itertools.product(['all', 'random', 'craig'], range(runs)):
        data = np.load(f'results/{setting}_{i}.npz')
        for item in data.files:
            if isinstance(eval(f"{setting}_data_raw['{item}']"), list):
                exec(f"{setting}_data_raw['{item}'].append(data['{item}'])")
            else:
                exec(f"{setting}_data_raw['{item}'] = [data['{item}']]")

    # (converting the above lists of numpy arrays into full numpy arrays)
    for key, _ in all_data_raw.items(): # note: same keys for all three dictionaries
        all_data_raw[key] = np.array(all_data_raw[key])
        random_data_raw[key] = np.array(random_data_raw[key])
        craig_data_raw[key] = np.array(craig_data_raw[key])

    return all_data_raw, random_data_raw, craig_data_raw


##############################################################################################
##############################################################################################
##############################################################################################


def generate_stats_primary(mode, raw_data):
    '''
    Compute and return the desired statistic (sample mean/variance/std. deviation) across the runs.

    Parameters
    - mode: str, one of ['mean', 'var', 'std'], the statistic that needs to be computed
    - raw_data: tuple[dict], raw data provided in the order (all, random, craig)

    Returns
    - all_data: dict, having computed the desired statistic for the 'all data'
    - random_data: dict, having computed the desired statistic for the 'random subsets'
    - craig_data: dict, having computed the desired statistic for the 'CRAIG subsets'
    '''

    all_data = {}
    random_data = {}
    craig_data = {}
    
    for key, _ in raw_data[0].items(): # note: same keys for all three dictionaries
        extra_command = ', ddof=1' if mode == 'var' or mode == 'std' else '' # for sample std/var
        all_data[key] = eval(f'np.{mode}(raw_data[0][key], axis=0{extra_command})')
        random_data[key] = eval(f'np.{mode}(raw_data[1][key], axis=0{extra_command})')
        craig_data[key] = eval(f'np.{mode}(raw_data[2][key], axis=0{extra_command})')

    return all_data, random_data, craig_data

    

##############################################################################################
##############################################################################################
##############################################################################################


def generate_graphs_primary(dependent, independent='epochs', epoch_number=15, class_number=0,
                            save_file=None, conduct_t_test=False, significance=0.05,
                            alternative='two-sided'):
    '''
    Generates graphs for the primary set of collected data (the 100 repeated trials).

    Parameters
    - dependent: str, one of ['test_loss', 'accuracy', 'epoch_duration', 
                              'class_accuracies', 'fpr', 'fnr']; the dependent (y) variable
    - independent: str, one of ['classes', 'epochs']; the independent (x) variable

    - epoch_number: int, from 1-15; (optional) the epoch number, if this is a control variable
    - class_number: int, from 0-9; (optional) the class number, if this is a control variable

    - save_file: str, (optional) the file to which the graph should be saved
        - if given as None, displays the graph
    
    - conduct_t_test: bool, whether a supplementary paired t-test should be conducted to
                            compare the 'CRAIG subsets' and 'all data' settings
    - significance: float, significance level for the t-test
    - alternative: str, one of ['lesser', 'greater', 'two-sided'];
                   type of alternative for the paired t-test

    Notes
    - Assumes that the data is coming from the way get_primary_results saves the files 
    '''

    
    # Load in the data and generate the mean and standard deviation data
    N = 100
    all_raw, random_raw, craig_raw = load_data_primary(N)
    all_mean, random_mean, craig_mean = generate_stats_primary('mean', (all_raw, random_raw, craig_raw))
    all_std, random_std, craig_std = generate_stats_primary('std', (all_raw, random_raw, craig_raw))

    # Set up the independent and dependent variables
    if independent == 'classes' and dependent in ['class_accuracies', 'fpr', 'fnr']:
        X = np.arange(10)
    else:
        # The independent variable is 'epochs'
        X = np.arange(15) + np.ones(15)

    # Set up the dependent variable, as well as the graph labels and title
    Y_index = '[dependent]'

    xlabel = independent.capitalize().replace('_', ' ')

    if dependent in ['fnr', 'fpr']:
        ylabel = dependent.upper().replace('_', ' ')
        unit = ''
    else:
        ylabel = dependent.capitalize().replace('_', ' ')
        unit = ' (seconds)' if dependent == 'epoch_duration' else ''

    title = f'Graph of {ylabel} vs. {xlabel}'    

    if dependent in ['class_accuracies', 'fpr', 'fnr']:
        # Indices of data required for the plot
        addition = '[epoch_number-1, :]' if independent == 'classes' else '[:, class_number]'
        Y_index += addition

        # Title modifications
        control_var = f' for epoch {epoch_number}' if independent == 'classes' else f' for class {class_number}'
        title += control_var

    # Necessary z-quantile for 99.9% central confidence interval using CLT
    z_quantile = stats.norm.interval(0.999)[1]

    # Plot the required graph (uses some clever string evaluation to obtain the
    # appropriate dataset and conf. interval for the Y-values from the full set of data above)
    _, ax = plt.subplots()

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel+unit, fontsize=12)
    ax.set_title(title, fontsize=16)

    ax.errorbar(X, eval(f'all_mean{Y_index}'),
                yerr=eval(f'z_quantile * all_std{Y_index} / np.sqrt(N)'), fmt='ro-', markersize=4, capsize=2, label='All data')

    ax.errorbar(X, eval(f'random_mean{Y_index}'),
                yerr=eval(f'z_quantile * random_std{Y_index} / np.sqrt(N)'), fmt='bo-',  markersize=4, capsize=2, label='Random subsets')

    ax.errorbar(X, eval(f'craig_mean{Y_index}'),
                yerr=eval(f'z_quantile * craig_std{Y_index} / np.sqrt(N)'), fmt='go-', markersize=4, capsize=2, label='CRAIG subsets')

    ax.legend()

    if save_file != None:
        plt.savefig(save_file)
    else:
        plt.show()


    # Conduct a series of two-sample t-test comparing CRAIG subsets (set X) and all data (set Y)
    # with a one-sided alternative
    if conduct_t_test:
        print('----------------------------------------------------')
        print(f'Two-sample t-test at {significance*100}% significance with {alternative} alternative')
        print(f'comparing CRAIG subsets to all training data.')
        print(f'Dependent: {dependent}, independent: {independent}, epoch: {epoch_number}, class: {class_number}')

        x_bar = eval(f'craig_mean{Y_index}')
        y_bar = eval(f'all_mean{Y_index}')
        n = m = N # here, 100 trials

        sx2 = eval(f'craig_std{Y_index}')**2
        sy2 = eval(f'all_std{Y_index}')**2

        sp2 = ((n-1)*sx2 + (m-1)*sy2) / (m+n-2)

        t = (x_bar - y_bar) / np.sqrt(sp2 * (1/m + 1/n))

        if alternative == 'lesser':
            t_quantile = stats.t.interval(1 - (significance * 2), df=m+n-2)[0] * np.ones_like(t)
            print(t < t_quantile)
        elif alternative == 'greater':
            t_quantile = stats.t.interval(1 - (significance * 2), df=m+n-2)[1] * np.ones_like(t)
            print(t > t_quantile)
        else:
            # 'two-sided' alternative hypothesis
            t_quantile = stats.t.interval(1 - significance, df=m+n-2)[1] * np.ones_like(t)
            print(np.logical_and(t < -1*t_quantile, t > t_quantile))



##############################################################################################
##############################################################################################
##############################################################################################


def load_data_extension(subset_sizes):
    '''
    Load in the data from the primary part of this investigation.

    Parameters
    - subset_sizes: list[float], the set of subset sizes for which the data should be loaded in

    Returns
    - all_data_raw: dict, contains all collected data for the 'all' setting
    - random_data_raw: dict, contains all collected data for the 'random' setting
    - craig_data_raw: dict, contains all collected data for the 'craig' setting

    Notes
    - Assumes that the data is coming from the way get_extension_results saves the files
    '''

    all_data_raw = {
        'test_loss': None,
        'accuracy': None,
        'class_accuracies': None,
        'fpr': None,
        'fnr': None,
        'epoch_duration': None
    }

    random_data_raw = {
        'test_loss': None,
        'accuracy': None,
        'class_accuracies': None,
        'fpr': None,
        'fnr': None,
        'epoch_duration': None
    }

    craig_data_raw = {
        'test_loss': None,
        'accuracy': None,
        'class_accuracies': None,
        'fpr': None,
        'fnr': None,
        'epoch_duration': None
    }

    # Cleverly extracts all the data from the 'extension_results' folder and collects them into
    # the above dictionaries
    for setting, subset_size in itertools.product(['random', 'craig'], subset_sizes):
        data = np.load(f'extension_results/{setting}_{subset_size}_0.npz')
        for item in data.files:
            if isinstance(eval(f"{setting}_data_raw['{item}']"), list):
                exec(f"{setting}_data_raw['{item}'].append( ({subset_size}, data['{item}']) )")
            else:
                exec(f"{setting}_data_raw['{item}'] = [ ({subset_size}, data['{item}']) ]")

    data = np.load('extension_results/all_0.npz')
    for item in data.files:
        all_data_raw[item] = data[item]

    return all_data_raw, random_data_raw, craig_data_raw



##############################################################################################
##############################################################################################
##############################################################################################


def generate_graphs_extension(dependent, independent='subset_size', 
                              epoch_number=50, class_number=0, subset_size=0.4,
                              save_file=None):
    '''
    Generates graphs for the extended set of collected data (changing subset-sizes)

    Parameters
    - dependent: str, one of ['test_loss', 'accuracy', 'epoch_duration', 
                                'class_accuracies', 'fpr', 'fnr']; the dependent (y) variable
    - independent: str, one of ['classes', 'epochs', 'subset_size']; the independent (x) variable

    - epoch_number: int, from 1-15; (optional) the epoch number, if this is a control variable
    - class_number: int, from 0-9; (optional) the class number, if this is a control variable
    - subset_size: float; (optional) the subset size, if this is a control variable

    - save_file: str, (optional) the file to which the graph should be saved
        - if given as None, displays the graph

    Notes
    - Assumes that the data is coming from the way get_extension_results saves the files 
    '''
    
    # Load in the data
    all_subset_sizes = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    ext_all, ext_random, ext_craig = load_data_extension(all_subset_sizes)

    # Set up the independent variable (control variables already set up)
    if independent == 'classes':
        X = np.arange(10)
        subset_size = all_subset_sizes.index(subset_size) if subset_size in all_subset_sizes else 2
    elif independent == 'epochs':
        X = np.arange(50) + np.ones(50)
        subset_size = all_subset_sizes.index(subset_size) if subset_size in all_subset_sizes else 2
    else:
        # The independent variable is 'subsets'
        X = np.array(all_subset_sizes)


    # Set up the dependent variable for each setting, as well as the graph labels and title
    xlabel = independent.capitalize().replace('_', ' ')

    if dependent in ['fnr', 'fpr']:
        ylabel = dependent.upper().replace('_', ' ')
        unit = ''
    else:
        ylabel = dependent.capitalize().replace('_', ' ')
        unit = ' (seconds)' if dependent == 'epoch_duration' else ''

    title = f'Graph of {ylabel} vs. {xlabel} for '

    if dependent in ['class_accuracies', 'fpr', 'fnr']:
        if independent == 'classes':
            Y_all = ext_all[dependent][epoch_number-1, :]
            Y_random = ext_random[dependent][subset_size][1][epoch_number-1, :]
            Y_craig = ext_craig[dependent][subset_size][1][epoch_number-1, :]

            control_vars = f'subset size {all_subset_sizes[subset_size]}, epoch {epoch_number}'

        elif independent == 'epochs':
            Y_all = ext_all[dependent][:, class_number]
            Y_random = ext_random[dependent][subset_size][1][:, class_number]
            Y_craig = ext_craig[dependent][subset_size][1][:, class_number]

            control_vars = f'subset size {all_subset_sizes[subset_size]}, class {class_number}'

        else:
            # The independent variable is 'subset_size'
            Y_all = ext_all[dependent][epoch_number-1, class_number] * np.ones_like(X)
            Y_random = [item[1][epoch_number-1, class_number] for item in ext_random[dependent]]
            Y_craig = [item[1][epoch_number-1, class_number] for item in ext_craig[dependent]]

            control_vars = f'epoch {epoch_number}, class {class_number}'

    else:
        # The dependent variable is 'test_loss', 'accuracy' or 'epoch_duration'
        if independent in ['classes', 'epochs']:
            Y_all = ext_all[dependent]
            Y_random = ext_random[dependent][subset_size][1]
            Y_craig = ext_craig[dependent][subset_size][1]

            control_vars = f'epoch {epoch_number}' if independent != 'epochs' else f'subset size {all_subset_sizes[subset_size]}'

        else:
            # The independent variable is 'subset_size'
            Y_all = ext_all[dependent][epoch_number-1] * np.ones_like(X)
            Y_random = [item[1][epoch_number-1] for item in ext_random[dependent]]
            Y_craig = [item[1][epoch_number-1] for item in ext_craig[dependent]]

            control_vars = f'epoch {epoch_number}'

    title += control_vars

    # Plot the required graph
    _, ax = plt.subplots()

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel+unit, fontsize=12)
    ax.set_title(title, fontsize=16)

    if independent != 'subset_size':
        ax.plot(X, Y_all, 'ro-', markersize=4, label='All data')
    else:
        ax.plot(X, Y_all, 'r--', label='All data')
    ax.plot(X, Y_random, 'bo-', markersize=4, label='Random subsets')
    ax.plot(X, Y_craig, 'go-', markersize=4, label='CRAIG subsets')

    ax.legend()
    
    if save_file != None:
        plt.savefig(save_file)
    else:
        plt.show()



##############################################################################################
##############################################################################################
##############################################################################################

def main():
    '''
    Main run loop.

    Notes
    - Uncomment as needed to run sections of the code rather than all at once.
    '''

    #get_primary_results()

    #get_extension_results()


    # Graphs from the primary set of data (100 trials)
    generate_graphs_primary('test_loss', independent='epochs',
                            save_file='graphs/primary_test_loss_epochs',
                            conduct_t_test=True, significance=0.001,
                            alternative='lesser')
    
    generate_graphs_primary('accuracy', independent='epochs',
                            save_file='graphs/primary_accuracy_epochs',
                            conduct_t_test=True, significance=0.001,
                            alternative='greater')
    
    generate_graphs_primary('epoch_duration', independent='epochs',
                            save_file='graphs/primary_epoch_duration_epochs')

    
    generate_graphs_primary('fnr', independent='classes',
                            save_file='graphs/primary_fnr_classes')
    generate_graphs_primary('fpr', independent='classes',
                            save_file='graphs/primary_fpr_classes')


    generate_graphs_primary('fnr', independent='epochs', class_number=0,
                            save_file='graphs/primary_fnr_0_epochs')
    generate_graphs_primary('fnr', independent='epochs', class_number=5,
                            save_file='graphs/primary_fnr_5_epochs')


    generate_graphs_primary('fpr', independent='epochs', class_number=0,
                            save_file='graphs/primary_fpr_0_epochs')
    generate_graphs_primary('fpr', independent='epochs', class_number=3,
                            save_file='graphs/primary_fpr_3_epochs')


    # Graphs from the extended set of results (changing subset size)
    generate_graphs_extension('test_loss', independent='subset_size', epoch_number=50,
                              save_file='graphs/extension_test_loss_subsets')
    generate_graphs_extension('accuracy', independent='subset_size', epoch_number=50,
                               save_file='graphs/extension_accuracy_subsets')
    generate_graphs_extension('epoch_duration', independent='subset_size', epoch_number=50,
                              save_file='graphs/extension_epoch_duration_subsets')


    generate_graphs_extension('fnr', independent='subset_size', class_number=0, epoch_number=50,
                              save_file='graphs/extension_fnr_0_50_subsets')
    generate_graphs_extension('fnr', independent='subset_size', class_number=5, epoch_number=50,
                              save_file='graphs/extension_fnr_5_50_subsets')

    
    generate_graphs_extension('fpr', independent='subset_size', class_number=0, epoch_number=50,
                              save_file='graphs/extension_fpr_0_50_subsets')
    generate_graphs_extension('fpr', independent='subset_size', class_number=3, epoch_number=50,
                              save_file='graphs/extension_fpr_3_50_subsets')


    generate_graphs_extension('fnr', independent='classes', epoch_number=15, subset_size=0.3,
                              save_file='graphs/extension_fnr_30pc_15_classes')
    generate_graphs_extension('fnr', independent='classes', epoch_number=50, subset_size=0.3,
                              save_file='graphs/extension_fnr_30pc_50_classes')

    
    generate_graphs_extension('fpr', independent='classes', epoch_number=15, subset_size=0.3,
                              save_file='graphs/extension_fpr_30pc_15_classes')
    generate_graphs_extension('fpr', independent='classes', epoch_number=50, subset_size=0.3,
                              save_file='graphs/extension_fpr_30pc_50_classes')
    

if __name__ == '__main__':
    main()