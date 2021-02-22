from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from TSX.utils import load_data, load_county_data, get_initial_county_data, train_model, load_county_name, get_importance_value, \
    plot_temporal_importance, get_top_importance_value, get_normalize_importance_value, get_hotspot_weight, \
    train_model_multitask, load_ckp, mean_absolute_percentage_error
from TSX.models import IMVTensorLSTM, IMVTensorLSTMMultiTask

import os
import sys
import argparse
import tqdm
import torch
import numpy as np
import pandas as pd
import pickle
import math
import timeit
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', weight='bold')


def train_one_county(fips, state, train_loader, valid_loader, ft_size):
    hidden_size = args.hidden_size
    n_epochs = args.n_epochs
    if args.explainer == 'IMVTensorLSTM':
        model = IMVTensorLSTM(ft_size, 1, hidden_size, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=1)
        if args.train:
            train_model(model, args.explainer, train_loader, valid_loader, optimizer=optimizer,
                        epoch_scheduler=epoch_scheduler, n_epochs=n_epochs,
                        device=device, county_fips_code=fips, state=state, cv=0)
        else:
            model_path = '../model_save/' + args.explainer + "/" + state + '/' + str(fips) + ".pt"
            model.load_state_dict(torch.load(model_path))
    else:
        model = []
    return model


def evaluate_performance(model, state, fips, county_name, scaler, X_train, y_train, X_test, y_test, X_total, y_total):
    scaler_for_cases = MinMaxScaler()
    scaler_for_cases.min_, scaler_for_cases.scale_ = scaler.min_[0], scaler.scale_[0]
    model.eval()
    if args.explainer == "IMVTensorLSTMMultiTask" or args.explainer == 'TransferLearning':
        X_total, task_idx, activated_share_columns = X_total
        # activated_share_columns = activated_share_columns[0, :]
        task_idx = task_idx.type(torch.LongTensor)
        with torch.no_grad():
            total_predict, total_alphas, total_betas, theta, neg_llk = model(X_total.to(device), y_total.to(device), task_idx, activated_share_columns)
            train_predict, train_alphas, train_betas, theta, neg_llk = model(X_train.to(device), y_train.to(device), task_idx, activated_share_columns)
            if state != "NY_flu":
                test_predict, test_alphas, test_betas, theta, neg_llk = model(X_test.to(device), y_test.to(device), task_idx, activated_share_columns)
    else:
        with torch.no_grad():
            total_predict, total_alphas, total_betas = model(X_total.to(device))
            train_predict, train_alphas, train_betas = model(X_train.to(device))
            test_predict, test_alphas, test_betas = model(X_test.to(device))
    total_predict_back = scaler_for_cases.inverse_transform(total_predict.detach().cpu().numpy())
    total_y_back = scaler_for_cases.inverse_transform(y_total.data.numpy())
    if state != "NY_flu":
        test_predict_back = scaler_for_cases.inverse_transform(test_predict.detach().cpu().numpy())
        test_y_back = scaler_for_cases.inverse_transform(y_test.data.numpy())
        mse = mean_squared_error(test_y_back, test_predict_back)
        mae = mean_absolute_error(test_y_back, test_predict_back)
        mape = mean_absolute_percentage_error(test_y_back, test_predict_back)
        rmse = round(np.sqrt(mse), 3)
        mae = round(mae, 3)
    else:
        mae = 0
        rmse = 0
        mape = 0
    alphas, betas = get_importance_value(train_alphas, train_betas)

    #print('County {} MAE: {}'.format(fips, mae))
    #print('County {} RMSE: {}'.format(fips, rmse))
    if state == "NY_flu":
        title = 'Daily flu Cases Prediction for {} county'.format(county_name)
    else:
        title = 'Daily covid Cases Prediction for {} county'.format(county_name)
    if args.save:
        plt.axvline(x=len(X_train) - 1, c='r', linestyle='--')
        plt.plot(total_y_back, color='blue', label='Real')
        plt.plot(total_predict_back, color='orange', label='Pred')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Number of cases')
        plt.legend()
        plot_path = '../plots/' + args.explainer + "/" + state + '/' + str(fips) + "/" + "time_series.pdf"
        plt.savefig(plot_path, dpi=300, orientation='landscape')
        #plt.show()
        plt.close()
    return mae, rmse, mape, alphas, betas


def state_level_computation(state):
    # Load Data
    path = '../data/state_mobility_link/' + state + '_mobility_link.csv'
    df = load_data(path)
    feature_county_list = list(df.columns[4:])
    county_list = list(set(df['next_area_fip'].tolist()))[9:10]
    county_list = [str(ct) for ct in county_list]
    county_name_list = load_county_name(county_list)
    county_dict = dict(zip(county_list, county_name_list))
    mae_list = []
    rmse_list = []
    mape_list = []
    valid_county_list = []
    county_importance_dict = dict(zip(['county_covid_cases'] + feature_county_list, np.zeros(len(feature_county_list) + 1)))
    error_county_list = []
    state_adjacency_matrix = np.zeros((len(county_list), len(county_list)))
    num = 0
    for county_fips, county_name in tqdm.tqdm(county_dict.items(), file=sys.stdout):
        county_plot_path = '../plots/' + args.explainer + "/" + state + '/' + str(county_fips)
        if not os.path.exists(county_plot_path):
            os.mkdir(county_plot_path)
        print("Computing for county {}, fips {}".format(county_name, county_fips))
        try:
            data_train_loader, data_test_loader, X_train, y_train, X_test, y_test, X_total, \
                 y_total, scaler, feature_name, feature_fips = \
                load_county_data(df, county_fips, args.seq_length, args.batch_size, args.test_data_size, args.fillna)
        except ValueError as err:
            print(err.args)
            error_county_list += [county_fips]
            continue
        feature_size = X_total.shape[2]
        model = train_one_county(county_fips, state, data_train_loader, data_test_loader, feature_size)
        mae, rmse, mape, alphas, betas = evaluate_performance(model, state, county_fips, county_name, scaler, X_train, y_train,
                                                       X_test, y_test, X_total, y_total)
        normalize_betas = get_normalize_importance_value(betas)
        hotspot_weight = get_hotspot_weight(y_train)
        for i, ct in enumerate(feature_fips):
            county_importance_dict[ct] += normalize_betas[i] * hotspot_weight

        # adjacency matrix
        for i, ct in enumerate(feature_fips):
            if ct in county_list:
                state_adjacency_matrix[num, county_list.index(ct)] = normalize_betas[i] * hotspot_weight
        num = num + 1
        alphas, betas, feature_name = get_top_importance_value(alphas, betas, feature_name)
        plot_temporal_importance(alphas, feature_name, args.explainer, state, county_fips, county_name)

        mae_list += [mae]
        rmse_list += [rmse]
        mape_list += [mape]
        valid_county_list += [county_fips]
    performance_df = pd.DataFrame({'county': valid_county_list, 'MAE': mae_list, 'RMSE': rmse_list, 'MAPE': mape_list})
    importance_df = pd.DataFrame(county_importance_dict.items(), columns=['fips', 'Importance_score']).\
        sort_values(by='Importance_score', ascending=False)
    importance_df['county_name'] = load_county_name(importance_df['fips'])
    print(performance_df.head())
    print(importance_df.head())
    output_path = "../outputs/" + args.explainer + "/" + state + "/"
    performance_path = output_path + "Total_county_performance.csv"
    importance_path = output_path + "Total_county_importance.csv"
    # performance_df.to_csv(performance_path)
    # importance_df.to_csv(importance_path)

    adjacency_df = pd.DataFrame(state_adjacency_matrix)
    adjacency_df.columns = county_name_list
    adjacency_df.index = county_list
    # adjacency_df.to_csv(output_path + "adjacency_matrix.csv")

    with open(output_path + "error_county.txt", "wb") as fp:  # Pickling
        pickle.dump(error_county_list, fp)


def state_level_computation_multitask(state, transfer):
    # Load Data
    if not transfer:
        path = '../data/state_mobility_link/' + state + '_mobility_link.csv'
        df = load_data(path)
    else:
        path = '../data/NY_flu_covid/'
        if state == "NY_flu":
            df = load_data(path + 'NY_flu_mobility_link_weekly.csv')
        else:
            df = load_data(path + 'NY_covid_mobility_link_weekly.csv')
    input_task_feature = args.input_task_feature
    input_task_feature_name = ['county_covid_cases', 'next_area_cmi']
    feature_county_list = list(df.columns[3 + input_task_feature:])
    county_list = list(set(df['next_area_fip'].tolist()))
    county_list = [17031, 17019, 17097, 17043, 17197]
    county_list = [str(ct) for ct in county_list]
    county_name_list = load_county_name(county_list)
    county_dict = dict(zip(county_list, county_name_list))
    error_county_list = []
    county_importance_dict = dict(
        zip(input_task_feature_name + feature_county_list, np.zeros(len(feature_county_list) + input_task_feature)))

    mae_list = []
    rmse_list = []
    mape_list = []
    valid_county_list = []
    data_train_loader_list = []
    data_test_loader_list = []
    print(county_name_list)
    task_num = 0
    for county_fips, county_name in county_dict.items():
        try:
            county_plot_path = '../plots/' + args.explainer + "/" + state + '/' + str(county_fips)
            if not os.path.exists(county_plot_path):
                os.mkdir(county_plot_path)

            _ = get_initial_county_data(df, county_fips, args.fillna, input_task_feature, task_num)
            task_num += 1
        except ValueError as err:
            print(err.args)
            error_county_list += [county_fips]
            continue
    task_idx = 0
    train_size = 0
    for county_fips, county_name in county_dict.items():
        try:
            data_train_loader, data_test_loader, X_train, y_train, X_test, y_test, X_total, \
                y_total, scaler, feature_name, feature_fips = \
                load_county_data(df, county_fips, args.seq_length, args.batch_size, args.test_data_size,
                                args.fillna, input_task_feature, input_task_feature_name, task_idx)
            task_idx += 1
            data_train_loader_list += [data_train_loader]
            data_test_loader_list += [data_test_loader]
            if X_train.shape[0] > train_size:
                train_size = X_train.shape[0]
        except ValueError as err:
            print(err)
            continue

    input_share_dim = len(feature_county_list)
    hidden_size = args.hidden_size
    n_epochs = args.n_epochs
    iterations = math.ceil(train_size / args.batch_size)
    start_time = timeit.default_timer()

    flu_model = IMVTensorLSTMMultiTask(input_share_dim, input_task_feature, task_num, 1, hidden_size, device, args.em, args.drop_prob).to(device)
    if transfer:
        if state == "NY_flu_covid":
            model_path = '../model_save/' + args.explainer + '/NY_flu/NY_flu.pt'
            flu_model.load_state_dict(torch.load(model_path))

    model = IMVTensorLSTMMultiTask(input_share_dim, input_task_feature, task_num, 1, hidden_size, device, args.em,
                                    args.drop_prob).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=1)
    start_epoch = 0
    valid_loss_min = 9999
    if not args.train:
        model_path = '../model_save/' + args.explainer + '/' + state + '/' + state + "_best.pt"
        model, optimizer, start_epoch, valid_loss_min = load_ckp(model_path, model, optimizer)
        print("model = ", model)
        print("optimizer = ", optimizer)
        print("start_epoch = ", start_epoch)
        print("valid_loss_min = {:.6f}".format(valid_loss_min))
        n_epochs = n_epochs - start_epoch
    if not args.evaluate:
        train_model_multitask(model, args.explainer, data_train_loader_list, data_test_loader_list, flu_model,
                              input_task_feature, start_epoch, valid_loss_min, optimizer=optimizer, epoch_scheduler=epoch_scheduler, n_epochs=n_epochs,
                              device=device, state=state, iterations=iterations, lambda_reg=args.lambda_reg, cv=0)

    stop_time = timeit.default_timer()
    task_idx = 0
    for county_fips, county_name in county_dict.items():
        try:
            data_train_loader, data_test_loader, X_train, y_train, X_test, y_test, X_total, \
                y_total, scaler, feature_name, feature_fips = \
                load_county_data(df, county_fips, args.seq_length, args.batch_size, args.test_data_size,
                                 args.fillna, input_task_feature, input_task_feature_name, task_idx)
        except ValueError as err:
            continue
        mae, rmse, mape, alphas, betas = evaluate_performance(model, state, county_fips, county_name, scaler, X_train,
                                                        y_train, X_test, y_test, X_total, y_total)
        for i, ct in enumerate(feature_fips):
            if i < input_task_feature:
                county_importance_dict[ct] += betas[input_task_feature*task_idx + i]
            else:
                county_importance_dict[ct] += betas[task_num*input_task_feature + i-input_task_feature]
        task_idx += 1
        mae_list += [mae]
        rmse_list += [rmse]
        mape_list += [mape]
        valid_county_list += [county_fips]
    performance_df = pd.DataFrame({'county': valid_county_list, 'MAE': mae_list, 'RMSE': rmse_list, 'MAPE': mape_list})
    importance_df = pd.DataFrame(county_importance_dict.items(), columns=['fips', 'Importance_score']). \
        sort_values(by='Importance_score', ascending=False)
    importance_df['county_name'] = load_county_name(importance_df['fips'])
    print(performance_df.head())
    print(importance_df.head(10))
    output_path = "../outputs/" + args.explainer + "/" + state + "/"
    performance_path = output_path + "Total_county_performance.csv"
    importance_path = output_path + "Total_county_importance.csv"
    if args.save:
        performance_df.to_csv(performance_path)
        importance_df.to_csv(importance_path)
    print('Training time: ', stop_time - start_time)


def main():
    # load full data list
    state_list = ["AK", "NY", "WA", "NV", "AZ", "AL", "FL", "GA", "MS", "TN", "MI", "AR", "LA", "MO", "OK", "TX",
                  "NM", "CA", "UT", "ND", "HI", "MN", "OR", "MT", "CO", "KS", "WY", "NE", "SD", "CT", "MA", "ME",
                  "VT", "RI", "MD", "VA", "DE", "PA", "OH", "NJ", "SC", "NC", "IA", "WI", "IL", "ID", "KY", "IN",
                  "WV", "NH", "DC"]
    state_list = ["IL"]
    for i, state in enumerate(state_list):
        if not os.path.exists('../plots/' + args.explainer + '/' + state):
            os.mkdir('../plots/' + args.explainer + '/' + state)
        if not os.path.exists('../model_save/' + args.explainer + '/' + state):
            os.mkdir('../model_save/' + args.explainer + '/' + state)
        if not os.path.exists('../outputs/' + args.explainer + '/' + state):
            os.mkdir('../outputs/' + args.explainer + '/' + state)
        print("Start for state {}, num {}".format(state, i))
        if args.explainer == "IMVTensorLSTMMultiTask":
            state_level_computation_multitask(state, 0)
        elif args.explainer == "IMVTensorLSTM":
            state_level_computation(state)
        elif args.explainer == "TransferLearning":
            state_level_computation_multitask(state, 1)


if __name__ == '__main__':
    np.random.seed(2021)
    parser = argparse.ArgumentParser(description='Run baseline model for covid')
    parser.add_argument('--explainer', type=str, default='IMVTensorLSTMMultiTask', help='Explainer model')
    parser.add_argument('--fillna', type=str, default='zero', help='fill na')
    parser.add_argument('--input_task_feature', type=int, default=2, help='input_task_feature')
    parser.add_argument('--seq_length', type=int, default=14, help='seq_length')
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden_size')
    parser.add_argument('--n_epochs', type=int, default=600, help='n_epochs')
    parser.add_argument('--test_data_size', type=int, default=1, help='test_data_size')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='drop prob')
    parser.add_argument('--lambda_reg', type=float, default=0.001, help='lambda regulation')
    parser.add_argument('--train', action='store_false')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--em', action='store_false')
    parser.add_argument('--save', action='store_false')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    if not os.path.exists('../plots'):
        os.mkdir('../plots')
    if not os.path.exists('../model_save'):
        os.mkdir('../model_save')
    if not os.path.exists('../outputs'):
        os.mkdir('../outputs')
    if not os.path.exists('../model_save/' + args.explainer):
        os.mkdir('../model_save/' + args.explainer)
    if not os.path.exists('../outputs/' + args.explainer):
        os.mkdir('../outputs/' + args.explainer)
    if not os.path.exists('../plots/' + args.explainer):
        os.mkdir('../plots/' + args.explainer)

    main()

    #for i in [0.1, 0.01, 0.001]:
    #    for j in [0.0, 0.1, 0.2]:
    #        args.lambda_reg = i
    #        args.drop_prob = j
    #        print("regulation: {}, drop prob: {}".format(i, j))
    #        main()



