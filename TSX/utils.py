import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_cenpop_dataset():
    cenpop = pd.read_csv("../data/county_population/CenPop2010_Mean_CO.txt",
                         sep=',', encoding='ISO-8859-1')
    cenpop = cenpop.append(dict(zip(cenpop.columns, [46, 102, 'Oglala Lakota', 'South Dakota', 13586, 43.33, -102.55])),
                           ignore_index=True)
    cenpop = cenpop.append(dict(zip(cenpop.columns, [2, 158, 'Kusilvak', 'Alaska', 7459, 62.09, -163.53])),
                           ignore_index=True)
    cenpop['area_id'] = cenpop.apply(lambda x: x['STATEFP'] * 1000 + x['COUNTYFP'], axis=1)
    return cenpop


def load_county_name(county_fips_list):
    cenpop = load_cenpop_dataset()
    feature_name = []
    for i, idx in enumerate(county_fips_list):
        if idx == 'county_covid_cases':
            feature_name += [idx]
        elif idx == 'next_area_cases':
            feature_name += ['county_covid_cases']
        elif idx == '36061':
            feature_name += ['NewYorkCity']
        else:
            feature_name += cenpop[cenpop['area_id'] == int(idx)]['COUNAME'].values.tolist()
    return feature_name


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def softmax(x):
    e = np.exp(x - np.max(x, axis=1).reshape((-1, 1)))
    return e / e.sum(axis=1).reshape((-1, 1))


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def load_data(path='./data/'):
    df = pd.read_csv(path)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


def get_initial_county_data(df, county_fips, fillna, task_idx):
    one_df = df[df["next_area_fip"] == int(county_fips)].sort_values("local_date")
    one_df = one_df[one_df.columns[3:]]
    if fillna == 'zero':
        one_df.fillna(value=0, inplace=True)
    elif fillna == 'ffill':
        one_df.iloc[0].fillna(0, inplace=True)
        one_df.fillna(method='ffill', inplace=True)
    criteria = one_df.sum(axis=0) != 0
    activated_shared_columns = np.array(criteria.astype(int).tolist()[1:])
    if task_idx == -1:
        one_df = one_df[criteria.index[criteria]]
    #if len(one_df.index) < 40:
    #    raise ValueError('Data samples too small')
    if len(criteria.index[criteria]) < 2:
        raise ValueError('feature samples too small')
    return one_df, activated_shared_columns


def load_county_data(df, county_fips, seq_length, batch_size, test_data_size, fillna, task_idx=-1):
    one_df, activated_shared_columns = get_initial_county_data(df, county_fips, fillna, task_idx)
    feature_name = load_county_name(one_df.columns)
    feature_fips = ['county_covid_cases'] + list(one_df.columns[1:])
    test_data = one_df[-test_data_size:]
    train_data = one_df[:-test_data_size]
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_data_normalized = scaler.fit_transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    total_data_normalized = scaler.transform(one_df)

    X_train, y_train = create_sequences(train_data_normalized, seq_length)
    X_test, y_test = create_sequences(test_data_normalized, seq_length)
    X_total, y_total = create_sequences(total_data_normalized, seq_length)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().view(-1, 1)

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().view(-1, 1)

    X_total = torch.from_numpy(X_total).float()
    y_total = torch.from_numpy(y_total).float().view(-1, 1)
    if task_idx == -1:
        data_train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size)
        data_test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size)
    else:
        task_train = torch.tensor(np.repeat(task_idx, X_train.shape[0]))
        task_test = torch.tensor(np.repeat(task_idx, X_test.shape[0]))
        task_train_activated_shared_columns = torch.tensor(np.repeat(activated_shared_columns[None, ], X_train.shape[0], axis=0), dtype=torch.long)
        task_test_activated_shared_columns = torch.tensor(np.repeat(activated_shared_columns[None, ], X_test.shape[0], axis=0), dtype=torch.long)
        data_train_loader = DataLoader(TensorDataset(X_train, y_train, task_train, task_train_activated_shared_columns), shuffle=False, batch_size=batch_size)
        data_test_loader = DataLoader(TensorDataset(X_test, y_test, task_test, task_test_activated_shared_columns), shuffle=False, batch_size=batch_size)
        X_total = [X_total, task_train, task_train_activated_shared_columns]
    return data_train_loader, data_test_loader, X_train, y_train, X_test, y_test, X_total, y_total, scaler, \
           feature_name, feature_fips


def train_model(model, model_name, train_loader, valid_loader, optimizer, epoch_scheduler, n_epochs,
                device, county_fips_code, state, cv=0):
    loss = torch.nn.MSELoss()
    patience = 80
    min_val_loss = 9999
    counter = 0
    save_path = '../model_save/' + model_name + '/' + state + '/'
    train_loss_trend = []
    test_loss_trend = []
    for i in range(n_epochs):
        mse_train = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            y_pred, alphas, betas = model(batch_x)
            # y_pred = y_pred.squeeze(1)
            l = loss(y_pred, batch_y)
            l.backward()
            mse_train += l.item()
            optimizer.step()
            epoch_scheduler.step()
        with torch.no_grad():
            mse_val = 0
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output_val, alphas_val, betas_val = model(batch_x)
                mse_val += loss(output_val, batch_y).item()
        train_loss_trend += [mse_train]
        test_loss_trend += [mse_val]
        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            # print("Saving...")
            # torch.save(model.state_dict(), save_path + str(county_fips_code) + ".pt")
            counter = 0
        else:
            counter += 1
        if counter == patience:
            break
        if i % 10 == 0:
            print("Iter: ", i, "train: ", mse_train ** 0.5, "val: ", mse_val ** 0.5)
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plot_path = '../plots/' + model_name + "/" + state + '/' + str(county_fips_code) + "/" + "loss_trend" + ".pdf"
    plt.savefig(plot_path)
    plt.close()


def train_model_multitask(model, model_name, data_train_loader_list, valid_loader_list, flu_model, input_task_feature,
                          optimizer, epoch_scheduler, n_epochs, device, state, iterations, lambda_reg, cv=0):
    loss_fn = torch.nn.MSELoss()
    patience = n_epochs
    min_val_loss = 9999
    counter = 0
    save_model_path = '../model_save/' + model_name + '/' + state + '/'
    train_loss_trend = []
    test_loss_trend = []
    data_train_loader_iterator_list = [iter(data_loader) for data_loader in data_train_loader_list]
    lambda_trans = 0.0
    for i in range(n_epochs):
        mse_train = 0
        for j in range(iterations):
            l_list = []
            optimizer.zero_grad()
            for k in range(len(data_train_loader_list)):
                try:
                    batch_x, batch_y, task_idx, activated_share_columns = next(data_train_loader_iterator_list[k])
                except StopIteration:
                    data_train_loader_iterator_list[k] = iter(data_train_loader_list[k])
                    batch_x, batch_y, task_idx, activated_share_columns = next(data_train_loader_iterator_list[k])
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                task_idx = task_idx[0].type(torch.LongTensor)
                activated_share_columns = activated_share_columns[0, :]
                y_pred, alphas, betas, theta, loss = model(batch_x, batch_y, task_idx, activated_share_columns)
                # l_list += [1 / (2 * torch.exp(theta)) * loss + theta / 2]
                l_list += [loss]
                regularization_params_list = ['F_alpha_n', 'F_alpha_n_b', 'F_beta.weight']
                for name, param in model.named_parameters():
                    if name in regularization_params_list:
                        # l_list += [1 / (2 * torch.exp(theta))
                        #           * torch.linalg.norm(param) * lambda_reg]
                        l_list += [torch.linalg.norm(param) * lambda_reg]
            if model_name == "TransferLearning" and state == "NY_flu_covid":
                flu_params = flu_model.state_dict()
                covid_params = model.state_dict()
                updating_params = list(covid_params.keys())[0:8]
                for name, param in model.named_parameters():
                    if name in updating_params:
                        l_list += [1 / (2 * torch.exp(theta))
                                   * torch.linalg.norm(param - flu_params[name]) * lambda_trans]
            l = sum(l_list)
            l.backward()
            mse_train += l.item()
            optimizer.step()
            epoch_scheduler.step()
        with torch.no_grad():
            mse_val = 0
            for valid_loader in valid_loader_list:
                for batch_x, batch_y, task_idx, activated_share_columns in valid_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    activated_share_columns = activated_share_columns[0, :]
                    output_val, alphas_val, betas_val, theta, loss = model(batch_x, batch_y, task_idx[0].type(torch.LongTensor), activated_share_columns)
                    mse_val += loss_fn(output_val, batch_y).item()
        train_loss_trend += [mse_train]
        test_loss_trend += [mse_val]
        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            counter = 0
        else:
            counter += 1
        if counter == patience:
            break
        if i % 100 == 0:
            print("Iter: ", i, "train: ", mse_train ** 0.5, "val: ", mse_val ** 0.5)
            torch.save(model.state_dict(), save_model_path + state + ".pt")
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plot_path = '../plots/' + model_name + "/" + state + "/" + "loss_trend" + ".pdf"
    plt.savefig(plot_path)
    plt.close()


def get_importance_value(alphas, beta):
    alphas = alphas.detach().cpu().numpy()
    betas = beta.detach().cpu().numpy()
    alphas = alphas.mean(axis=0)
    betas = betas.mean(axis=0)
    alphas = alphas[..., 0]
    betas = betas[..., 0]
    alphas = alphas.transpose(1, 0)
    return alphas, betas


def get_hotspot_weight(y_train):
    y_train = y_train.cpu().detach().numpy()
    if sum(y_train) != 0:
        return (sum(y_train[-3:]) / sum(y_train))[0]
    else:
        return 0


def get_top_importance_value(alphas, beta, feature_name, n=10):
    if len(feature_name) < n:
        n = len(feature_name)
    zipped = list(zip(feature_name, beta, alphas))
    top_feature = sorted(zipped, key=lambda x: x[1], reverse=True)[0:n]
    feature_name, beta, alphas = zip(*top_feature)
    return np.array(alphas), np.array(beta), feature_name


def get_normalize_importance_value(betas):
    betas = betas - min(betas)
    normalize_betas = betas / np.linalg.norm(betas, ord=1)
    return normalize_betas


def get_overall_importance_value():
    pass


def plot_temporal_importance_sns(alphas, feature_name, explainer, fips, county_name):
    fig, ax = plt.subplots(figsize=(30, 30))
    ax = sns.heatmap(alphas, cmap="Reds", linewidths=.1, annot=True, fmt=".2f")
    plt.show()


def plot_temporal_importance(alphas, feature_name, explainer, state, fips, county_name):
    plt.rcParams["axes.grid"] = False
    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(alphas, cmap='RdPu', vmax=np.max(alphas))
    ax.set_xticks(np.arange(alphas.shape[1]))
    ax.set_yticks(np.arange(len(feature_name)))
    ax.set_xticklabels(["t-" + str(i) for i in np.arange(alphas.shape[1], 0, -1)])
    ax.set_yticklabels(list(feature_name))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)
    for i in range(alphas.shape[0]):
        for j in range(alphas.shape[1]):
            text = ax.text(j, i, round(alphas[i, j], 3),
                           ha="center", va="center", color="w")
    ax.set_title("Importance of features and timesteps for county {}".format(county_name))
    # fig.tight_layout()
    plot_path = '../plots/' + explainer + "/" + state + '/' + str(fips) + "/" + "temporal_importance.pdf"
    plt.savefig(plot_path, orientation='Portrait')
    #plt.show()
    plt.close()



