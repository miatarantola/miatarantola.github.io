import geopandas as gpd
import folium
import mapclassify
import matplotlib.pyplot as plt
import pandas as pd

class Mapping: 
                              
    def plot_df(self, df, column): 
        from urllib.request import urlopen
        import plotly.io as pio
        import json
        with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
            counties = json.load(response)
        import plotly.express as px
        
        #load county data
        geo_data = gpd.read_file(
            'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')
        geo_data.id = geo_data.id.astype(str).astype(int)

        #shifting state/county codes from objects to integers

        geo_data['COUNTY'] = geo_data.COUNTY.astype(str).astype(int)
        geo_data['STATE'] = geo_data.STATE.astype(str).astype(int)
        
        #merge geodata and df
        merged_df = pd.merge(geo_data, df, left_on='COUNTY', right_on='County Code')
        merged_df["black_alone_percent"] = merged_df["Black Alone (F) %"] + merged_df["Black Alone (M) %"]

        
        merged_df["id"] = merged_df.apply(lambda row: str(row["id"]).zfill(5),axis=1)
        
        fig = px.choropleth_mapbox(merged_df, geojson=counties, locations='id',color=column,
                                   color_continuous_scale="Viridis",
                                   
                                   mapbox_style="carto-positron",
                                   zoom=2.75, center = {"lat": 37.0902, "lon": -95.7129},
                                   
                                   opacity=0.5,
                                  )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        pio.renderers.default = 'iframe'
        fig.show()

        
        