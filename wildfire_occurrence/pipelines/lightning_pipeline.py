import os
import re
import cudf
import cuml
import random
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd

from glob import glob
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from shapely.geometry import Point
from cuml.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from cupy import asnumpy
from joblib import dump, load
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split as sk_train_test_split
import xgboost as xgb

# ----------------------------------------------------

import logging

from wildfire_occurrence.config import Config
from wildfire_occurrence.common import read_config, get_filenames


class LightningPipeline(object):

    def __init__(
                self,
                config_filename: str,
            ):

        # Configuration file intialization
        self.conf = read_config(config_filename, Config)
        logging.info(f'Loaded configuration from {config_filename}')

        # outline the steps
        # download ALDN
        # find tif files to use to training generation
        # generate hit and miss dataset
        # train using this dataset
        # predict using rasters
        # validate using this dataset
        # ---- validate with dates not seen in training
        # predict the tundra
        # predict the entire alaska

        # data_regex: Lisstr,
        # label_regex: str = None,
        # aoi_regex: str = None,
        # output_filename: str = None

        # what is this??????
        self.delta_day = 0
        self.delta_hour = 3

        # aoi shapefile
        # self.aoi_regex = aoi_regex
        # self.aoi_gdf = gpd.read_file(aoi_regex)
        self.aoi_gdf = gpd.read_file('/explore/nobackup/projects/ilab/projects/LobodaTFO/data/Geometries/Alaska_Tiger/tl_2018_02_anrc.shp')

        self.output_filename = 'my_database.gpkg' #output_filename

    def preprocess(self):

        # Working on the preprocessing of the project
        logging.info('Starting preprocess pipeline step')

        # Create output directory
        os.makedirs(self.conf.lightning_data_dir, exist_ok=True)

        # Get data filenames
        data_filenames = get_filenames(self.conf.data_regex_list)
        logging.info(f'Found {len(data_filenames)} data filenames.')

        # Get label filenames
        label_filenames = get_filenames(self.conf.label_regex_list)
        logging.info(f'Found {len(label_filenames)} label filenames.')

        # Open label filenames, think of a better way
        label_epoch1_gdf = gpd.read_file(label_filenames[0])
        label_epoch2_gdf = gpd.read_file(label_filenames[1])

        # select the timestamps we want
        # we want time 0, time + 24, time + 48, etc.

        # variable to store full dataset
        full_dataset = []

        # iterate over each file
        for filename in data_filenames:

            # logging.info(f'Processing {filename}')

            # get datetime from WRF
            date_m = re.search(
                r'(?P<Y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})_' +
                r'(?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})', filename)
            # print("date_m", date_m)

            # convert date to datatime object
            datetime_str = \
                f'{date_m["Y"]}/{date_m["m"]}/{date_m["d"]} ' + \
                f'{date_m["H"]}:{date_m["M"]}:{date_m["S"]}'
            start_date = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')
            # print("start_date", start_date)

            # get the start date in the after time
            # start_date = start_date + timedelta(
            #        days=self.delta_day, hours=self.delta_hour)

            # check if the our is the beginning of the day
            if start_date.hour == 0:

                logging.info(f'Getting points from: {filename}')

                # get delta time + 1 day
                end_date = start_date + timedelta(
                    days=self.delta_day, hours=self.delta_hour)
                # print("end_date", end_date)
                # print(start_date, end_date)
                # print(start_date, end_date, filename)

                if start_date.year < 2012:
                    # get lightning rows
                    lightining_gdf = label_epoch1_gdf[
                        (label_epoch1_gdf['STRIKETIME'] > start_date) &
                        (label_epoch1_gdf['STRIKETIME'] < end_date)
                    ].reset_index(drop=True)
                else:
                    # get lightning rows
                    lightining_gdf = label_epoch2_gdf[
                        (label_epoch2_gdf['UTCDATETIME'] > start_date) &
                        (label_epoch2_gdf['UTCDATETIME'] < end_date)
                    ].reset_index(drop=True)
                    lightining_gdf = lightining_gdf[
                        lightining_gdf['STROKETYPE'] == 'GROUND_STROKE']

                print(start_date, end_date, lightining_gdf.shape)

                # if we do not have any lightning events, continue
                if lightining_gdf.shape[0] == 0:
                    continue

                output_lightning_filename = os.path.join(
                    self.conf.lightning_data_dir,
                    f'{Path(filename).stem}-lightning.gpkg'
                )

                lightining_gdf.to_file(
                    output_lightning_filename,
                    driver='GPKG', layer='lightning')

                # open raster with rioxarray and rasterio
                raster = rxr.open_rasterio(filename)
                rasterio_src = rasterio.open(filename)

                # convert crs from lightning to match crs from WRF
                lightining_gdf = lightining_gdf.to_crs(rasterio_src.crs)
                aoi_gdf = self.aoi_gdf.to_crs(rasterio_src.crs)

                # ------------------------------------------------------------
                # HERE WE SELECT TRUE HITS
                # ------------------------------------------------------------
                hits_coord_list = []
                for x, y in zip(
                            lightining_gdf['geometry'].x,
                            lightining_gdf['geometry'].y
                        ):
                    hits_coord_list.append((x, y))

                # get wrf values from true lightining hits
                hits_values = [x for x in rasterio_src.sample(hits_coord_list)]

                # create dataframe with hits and no hits
                hits_df = pd.DataFrame(
                    np.array(hits_values),
                    columns=list(raster.attrs['long_name'])
                ).reset_index(drop=True)

                # combine lightning data with WRF hits data, add label column
                hits_lightining_gdf = pd.concat(
                    [lightining_gdf, hits_df], axis=1)
                hits_lightining_gdf['Label'] = 1

                # remove rows with no-data values on WRF data
                # print("BEFORE REMOVING NODATA", hits_lightining_gdf.shape)
                hits_lightining_gdf = hits_lightining_gdf.dropna(
                    subset=list(raster.attrs['long_name']))
                print("AFTER REMOVING NODATA", hits_lightining_gdf.shape)

                hits_lightining_gdf = hits_lightining_gdf[
                    hits_lightining_gdf['geometry'].notna()]

                if hits_lightining_gdf.shape[0] == 0:
                    continue

                # ------------------------------------------------------------
                # HERE WE SELECT FALSE HITS
                # ------------------------------------------------------------
                no_hits_coord_list = []
                no_hits_values = []

                n_false_points = 0
                raster_x_coords = raster.coords['x'].values
                raster_y_coords = raster.coords['y'].values

                while n_false_points < hits_lightining_gdf.shape[0]:# + 10: #600
                    coord = (
                        raster_x_coords[
                            random.randint(0, raster_x_coords.shape[0] - 1)],
                        raster_y_coords[
                            random.randint(0, raster_y_coords.shape[0] - 1)],
                    )

                    if coord not in hits_coord_list and \
                            aoi_gdf.contains(Point(coord)).any():
                    # if coord not in hits_coord_list:
                        for wrf_value in rasterio_src.sample([coord]):
                            if not np.isnan(wrf_value).any():
                                no_hits_coord_list.append(coord)
                                no_hits_values.append(wrf_value)
                                n_false_points += 1

                # create dataframe with hits and no hits
                no_hits_df = pd.DataFrame(
                    np.array(no_hits_values),
                    columns=list(raster.attrs['long_name'])
                ).reset_index(drop=True)

                # add geometry to no_hits_df and combine with overall gdf
                no_hits_df_coords = pd.DataFrame.from_records(
                    no_hits_coord_list, columns=['x', 'y'])
                no_hits_lightining_gdf = gpd.GeoDataFrame(
                    no_hits_df, geometry=gpd.points_from_xy(
                        no_hits_df_coords.x, no_hits_df_coords.y),
                    crs=rasterio_src.crs
                )
                no_hits_lightining_gdf['Label'] = 0

                # print(no_hits_lightining_gdf.columns)
                # print("NO HITS", type(no_hits_lightining_gdf),
                # no_hits_lightining_gdf.shape)
                # print(list(hits_lightining_gdf.columns))
                # print(list(no_hits_lightining_gdf.columns))

                if 'geometry' not in list(hits_lightining_gdf.columns):
                    print("AHHHHHHHHHHHHHHHHH")

                if 'geometry' not in list(no_hits_lightining_gdf.columns):
                    print("AAHAAHHAHAHAHAHHAHAHAHAH")

                # append concatenation to full dataset
                concatenated = pd.concat(
                    [hits_lightining_gdf, no_hits_lightining_gdf],
                    ignore_index=True, axis=0)
                # print(concatenated.columns)
                full_dataset.append(
                    concatenated
                )

        full_dataset = pd.concat(full_dataset, axis=0)

        original_columns = full_dataset.columns

        print(full_dataset[full_dataset['geometry'].isna()])

        # print(full_dataset)
        full_dataset = full_dataset.dropna(axis='columns')
        # post_columns = full_dataset.columns
        # print(list(set(original_columns) - set(post_columns)))
        print("THE SIZE OF MY FINAL", full_dataset.shape)

        # print(full_dataset.columns)
        # print(type(full_dataset))
        full_dataset.to_file(
            self.output_filename, driver='GPKG', layer='dataset')

        return

    def train(self, dataset, model_filename):

        # read training dataset
        dataset = gpd.read_file(dataset)
        print("Dataset shape: ", dataset.shape)

        # drop geometry and bands with nodata
        dataset = dataset.dropna(axis='columns')
        dataset = dataset.drop(['geometry'], axis='columns').astype('float32')
        #dataset = cudf.from_pandas(dataset).astype('float32')
        # print(dataset.columns)

        # shuffle dataset, and split between training and validation
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        # random forest depth and size
        # We set the number of trees to 500 and the
        # number of variables at each split as 8 in the RF algorithm.
        n_estimators = 500
        max_depth = 10

        X_train, X_test, y_train, y_test = sk_train_test_split(
            dataset.drop(['Label'], axis='columns'),
            dataset['Label'],
            random_state=0,
            train_size=0.70,
            stratify=dataset['Label']  # TODO: double check what this means
        )

        #model = cuRF(
        #    max_depth=max_depth,
        #   n_estimators=n_estimators,
        #    random_state=0,
        #    n_streams=1
        #)



        hyperparameters = {'objective': 'binary:logistic',
                   'n_estimators':500,
                   'base_score': None,
                   'booster': 'gbtree',
                   'colsample_bylevel': None,
                   'colsample_bynode': None,
                   'colsample_bytree': None,
                   'gamma': None,
                   'gpu_id': None,
                   'interaction_constraints': None,
                   'learning_rate': 0.1,
                   'max_delta_step': None,
                   'max_depth': None,
                   'min_child_weight': None,
                   'monotone_constraints': None,
                   'n_jobs': -1,
                   'num_parallel_tree': None,
                   'random_state': None,
                   'reg_alpha': None,
                   'reg_lambda': None,
                   'scale_pos_weight': None,
                   'subsample': None,
                   #'tree_method': '',
                   'validate_parameters': None,
                   'verbosity': None
        }

        model = xgb.XGBClassifier(**hyperparameters)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        eval_metric = ["error","auc"]

        #model = RandomForestClassifier(
        #    max_depth=max_depth,
        #    n_estimators=n_estimators,
        #    random_state=0,
        #    #n_streams=1
        #)

        #trained_RF = model.fit(X_train, y_train)
        trained_RF = model.fit(X_train, y_train, eval_set=eval_set, eval_metric=eval_metric, early_stopping_rounds=10)

        predictions = model.predict(X_test)

        #cu_score = cuml.metrics.accuracy_score(y_test, predictions)
        #sk_score = accuracy_score(asnumpy(y_test), asnumpy(predictions))

        cu_score = accuracy_score(y_test, predictions)
        sk_score = accuracy_score(y_test, predictions)

        print(" cuml accuracy: ", cu_score)
        print(" sklearn accuracy : ", sk_score)
        print(model.feature_importances_)

        # save
        dump(trained_RF, model_filename)

        return

    def sliding_window_predict(self, data, model, ws=[5120, 5120]):
        """
        Predict from model.
        :param data: raster xarray object
        :param model: loaded model object
        :param ws: window size to predict on
        :return: prediction output in numpy format
        ----------
        Example
            raster.toraster(filename, raster_obj.prediction, outname)
        ----------
        """
        # open rasters and get both data and coordinates
        rast_shape = data[0, :, :].shape  # shape of the wider scene
        wsx, wsy = ws[0], ws[1]  # in memory sliding window predictions

        # if the window size is bigger than the image, predict full image
        if wsx > rast_shape[0]:
            wsx = rast_shape[0]
        if wsy > rast_shape[1]:
            wsy = rast_shape[1]

        prediction = np.zeros(rast_shape)  # crop out the window
        #print(f'wsize: {wsx}x{wsy}. Prediction shape: {prediction.shape}')

        for sx in range(0, rast_shape[0], wsx):  # iterate over x-axis
            for sy in range(0, rast_shape[1], wsy):  # iterate over y-axis
                x0, x1, y0, y1 = sx, sx + wsx, sy, sy + wsy  # assign window
                if x1 > rast_shape[0]:  # if selected x exceeds boundary
                    x1 = rast_shape[0]  # assign boundary to x-window
                if y1 > rast_shape[1]:  # if selected y exceeds boundary
                    y1 = rast_shape[1]  # assign boundary to y-window

                window = data[:, x0:x1, y0:y1]  # get window
                window = window.stack(z=('y', 'x'))  # stack y and x axis
                window = window.transpose("z", "band").values  # reshape

                #print(model.predict_proba(window).shape)

                # perform sliding window prediction
                # replace with predict proba
                prediction[x0:x1, y0:y1] = \
                    model.predict_proba(window)[:, 1].reshape((x1 - x0, y1 - y0))
        # save raster
        return prediction.astype('float32')  # type to int16

    def predict(
                self,
                model_filename: str,
                data_regex: str,
                deltatime: str = 'all',
                output_dir: str = None,
                aoi_gdf: str = None
            ):

        # deltatime can be a specify date, or all dates in the data regex
        os.makedirs(output_dir, exist_ok=True)

        # to reload the model uncomment the line below
        model = load(model_filename)

        # get data filenames
        data_filenames = sorted(glob(data_regex))

        for filename in data_filenames:

            # get datetime from WRF
            date_m = re.search(
                r'(?P<Y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})_' +
                r'(?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})', filename)

            datetime_str = \
                f'{date_m["Y"]}/{date_m["m"]}/{date_m["d"]} ' + \
                f'{date_m["H"]}:{date_m["M"]}:{date_m["S"]}'
            start_date = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')

            if start_date.hour == 0:

                # get delta time + 1 day
                # end_date = start_date + timedelta(days=1)
                # print(start_date, end_date, filename)
                output_filename = os.path.join(
                    output_dir, f'{Path(filename).stem}.lightning.tif')

                raster = rxr.open_rasterio(filename)
                raster = raster.fillna(0)
                #print(raster.shape)

                prediction = self.sliding_window_predict(
                    raster, model, ws=[5120, 5120])
                #print(prediction.shape)

                # Drop image band to allow for a merge of mask
                raster = raster.drop(
                    dim="band",
                    labels=raster.coords["band"].values[1:],
                )
                #print(raster.shape)

                # Get metadata to save raster
                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=0),
                    name='lightning',
                    coords=raster.coords,
                    dims=raster.dims,
                    attrs=raster.attrs
                )

                # Add metadata to raster attributes
                prediction.attrs['long_name'] = ('lightning_mask')
                prediction.attrs['model_name'] = (model_filename)
                #prediction = prediction.transpose("band", "y", "x")

                # Set nodata values on mask
                nodata = prediction.rio.nodata
                prediction = prediction.where(raster != nodata)
                prediction.rio.write_nodata(
                    nodata, encoded=True, inplace=True)

                prediction = prediction.rio.reproject("EPSG:3338")
                prediction = prediction.rio.clip(aoi_gdf.geometry.values, aoi_gdf.crs, drop=False, invert=False)


                # Save output raster file to disk
                prediction.rio.to_raster(
                    output_filename,
                    BIGTIFF='IF_SAFER',
                    compress='LZW',
                    driver='GTiff',
                    dtype='float32'
                )
                del prediction


        # predict_proba

        return

    def validation(self, model_filename, dataset_filename):

        # read model
        model = load(model_filename)

        # read dataset
        dataset = gpd.read_file(dataset_filename)#.sample(n = 40000)
        print(dataset)

        # run prediction and confusion matrix
        predictions = model.predict(dataset.drop(['Label', 'geometry'], axis=1))
        prediction_probabilities = model.predict_proba(dataset.drop(['Label', 'geometry'], axis=1))
        print(predictions)

        print(accuracy_score(dataset['Label'], predictions))
        print(confusion_matrix(dataset['Label'], predictions))

        conf_matrix = confusion_matrix(dataset['Label'], predictions)
        a = conf_matrix[0][0]
        b = conf_matrix[0][1]
        c = conf_matrix[1][0]
        d = conf_matrix[1][1]

        print(a, b, c, d)

        print("POD: ", a / (a+c))
        print("CSI: ", a / (a+b+c))
        print("FAR: ", b / (a+c))
        print("F:   ", b / (b+d))
        print("Brier: ", brier_score_loss(dataset['Label'], prediction_probabilities[:, 1]))
        print("Log Loss: ", log_loss(dataset['Label'], prediction_probabilities[:, 1]))



# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Generate Dataset
    data_regex = '/explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/*/*.tif'
    label_regex = [
        '/explore/nobackup/projects/ilab/projects/LobodaTFO/data/Alaska_Historical_Lightning/Tundra_Historical_Lightning_1986_2012_ImpactSystem_AlaskaAlbersNAD83.gpkg',
        '/explore/nobackup/projects/ilab/projects/LobodaTFO/data/Alaska_Historical_Lightning/Tundra_Historical_Lightning_2012_2022_TOA_AlaskaAlbersNAD83.gpkg'
    ]
    #aoi_regex = '/explore/nobackup/projects/ilab/projects/LobodaTFO/data/Geometries/Alaskan_Tundra/Alaska_tundra_merged.shp'
    aoi_regex = '/explore/nobackup/projects/ilab/projects/LobodaTFO/data/Geometries/Alaska_Tiger/tl_2018_02_anrc.shp'
    output_filename = '/explore/nobackup/projects/ilab/projects/LobodaTFO/labels/Lightning_Training_Dataset_New_Balanced_Real_24h-AllAlaska.gpkg'

    # initialize ligthning object
    lightning_model = LightningPipeline(
        #data_regex=data_regex,
        #label_regex=label_regex,
        #aoi_regex=aoi_regex,
        #output_filename=output_filename
        '/explore/nobackup/people/jacaraba/development/wildfire-occurrence/wildfire_occurrence/templates/config.yaml'
    )

    # lightning_model_2023-05-08_balanced_real


    # generate tabular dataset
    # lightning_model.preprocess()

    """
    # Train RF Model
    # lets train the RF model
    lightning_model.train(
        dataset='/explore/nobackup/projects/ilab/projects/LobodaTFO/labels/Lightning_Training_Dataset_New_Balanced_Real_24h-AllAlaska.gpkg',
        model_filename='/explore/nobackup/projects/ilab/projects/LobodaTFO/labels/LightningModel_Balanced_Real_24h_AllAlaska.sav'
    )
   
    # Predict with RF Model
    # given a date, look for the timestep and predict
    output_dir = '/explore/nobackup/projects/ilab/projects/LobodaTFO/development/lightning_model_2023-05-08_balanced_real_24h'

    lightning_model.predict(
        model_filename=model_filename,
        data_regex=data_regex,
        output_dir=output_dir
    )
    """

    #output_filename = '/explore/nobackup/projects/ilab/projects/LobodaTFO/data/Geometries/Alaska_Tiger/my_database.gpkg'
    #lightning_model.validation(
    #    model_filename=model_filename,
    #    dataset_filename=output_filename
    #)

    
    data_regex = '/explore/nobackup/projects/ilab/projects/LobodaTFO/operations/2023-06-29_2023-07-09/variables/*.tif'

    """
    # random forest
    model_filename = '/explore/nobackup/projects/ilab/projects/LobodaTFO/labels/LightningModel_Balanced_Real.sav'
    output_dir = '/explore/nobackup/projects/ilab/projects/LobodaTFO/operations/2023-06-29_2023-07-09/lightning-rf'

    lightning_model.predict(
        model_filename=model_filename,
        data_regex=data_regex,
        output_dir=output_dir,
        aoi_gdf=gpd.read_file(aoi_regex)
    )

    # xgboost
    model_filename = '/explore/nobackup/projects/ilab/projects/LobodaTFO/labels/LightningModel_Balanced_Real_24h.sav'
    output_dir = '/explore/nobackup/projects/ilab/projects/LobodaTFO/operations/2023-06-29_2023-07-09/lightning-xgboost'

    lightning_model.predict(
        model_filename=model_filename,
        data_regex=data_regex,
        output_dir=output_dir,
        aoi_gdf=gpd.read_file(aoi_regex)
    )
    """

    model_filename = '/explore/nobackup/projects/ilab/projects/LobodaTFO/labels/LightningModel_Balanced_Real_24h_AllAlaska.sav'
    output_dir = '/explore/nobackup/projects/ilab/projects/LobodaTFO/operations/2023-06-29_2023-07-09/lightning-xgboost-alaska'

    lightning_model.predict(
        model_filename=model_filename,
        data_regex=data_regex,
        output_dir=output_dir,
        aoi_gdf=gpd.read_file(aoi_regex)
    )