import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic

# calendar_df = pd.read_csv("data/calendar.csv")
data = pd.read_csv("data/listings.csv")


def clean_currency(x):
    if isinstance(x, str):
        return x.replace("$", "").replace(",", "")
    return x


data["target_price"] = data.price.apply(clean_currency).astype(float)


class PriceCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, price_columns, property_type_columns):
        self.price_columns = price_columns
        self.property_type_columns = property_type_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.price_columns:
            X[col] = X[col].astype(str).str.replace("$", "").str.replace(",", "").astype(float)
        for col in self.property_type_columns:
            X[col] = X[col].apply(self.clean_property_types).str.capitalize()

        return X

    @staticmethod
    def clean_property_types(prop_type: str):
        if prop_type.startswith("Entire"):
            prop_type = prop_type.replace("Entire ", "")
            return prop_type
        if prop_type.startswith("Private room in"):
            prop_type = prop_type.replace("Private room in ", "")
            return prop_type
        if prop_type.startswith("Shared room in"):
            prop_type = prop_type.replace("Shared room in ", "")
            return prop_type
        if prop_type.startswith("Room in"):
            prop_type = prop_type.replace("Room in ", "")
            return prop_type
        return prop_type


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cx_lat_lon = (51.5081, -0.1248)
        X["cx_distance_km"] = X.apply(
            lambda row: geodesic(cx_lat_lon, (row["latitude"], row["longitude"])).kilometers, axis=1
        )

        return X


class LocationClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, lat_col="latitude", lon_col="longitude"):
        self.n_clusters = n_clusters
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42)

    def fit(self, X, y=None):
        coords = X[[self.lat_col, self.lon_col]].copy()
        self.kmeans.fit(coords)
        return self

    def transform(self, X):
        X = X.copy()
        coords = X[[self.lat_col, self.lon_col]]
        X["location_cluster"] = self.kmeans.predict(coords).astype(str)
        return X


host_features = ["host_is_superhost"]
property_features = [
    "room_type",
    "accommodates",
]
review_features = ["review_scores_rating"]
geo_features = ["latitude", "longitude"]

host_pipeline = Pipeline(
    ("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))
)

property_pipeline = ColumnTransformer(
    [
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            ["room_type"],
        ),
        (
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
            ["accommodates"],
        ),
    ]
)

review_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    [
        ("host", host_pipeline, host_features),
        ("property", property_pipeline, property_features),
        ("review", review_pipeline, review_features),
    ]
)
