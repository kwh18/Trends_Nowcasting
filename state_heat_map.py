import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import geopandas as gpd
from shapely.geometry import Polygon
import os


def scrape_performance(states, perf_path):
    arma_09_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    arma_09_19 = pd.DataFrame(index=range(-30,66,5), columns= states)
    arma_21_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_09_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_09_19 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_21_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_joint_09_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_joint_09_19 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_joint_21_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_09_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_09_19 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_21_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_joint_09_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_joint_09_19 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_joint_21_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_all_09_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_all_09_19 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_all_21_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_all_joint_09_23 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_all_joint_09_19 = pd.DataFrame(index=range(-30,66,5), columns= states)
    dfm_mini_local_trends_all_joint_21_23 = pd.DataFrame(index=range(-30,66,5), columns= states)

    for state in states:
        arma_09_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local/performance/Model_performance_mini_{state.lower()}na_local_200901--202312.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=1).T
        arma_09_19[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local/performance/Model_performance_mini_{state.lower()}na_local_200901--201912.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=1).T
        arma_21_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local/performance/Model_performance_mini_{state.lower()}na_local_202101--202312.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=1).T
        dfm_mini_local_09_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local/performance/Model_performance_mini_{state.lower()}na_local_200901--202312.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        dfm_mini_local_09_19[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local/performance/Model_performance_mini_{state.lower()}na_local_200901--201912.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        dfm_mini_local_21_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local/performance/Model_performance_mini_{state.lower()}na_local_202101--202312.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_joint_09_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_joint/performance/Model_performance_mini_{state.lower()}na_local_joint_200901--202312.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_joint_09_19[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_joint/performance/Model_performance_mini_{state.lower()}na_local_joint_200901--201912.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_joint_21_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_joint/performance/Model_performance_mini_{state.lower()}na_local_joint_202101--202312.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        dfm_mini_local_trends_09_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends/performance/Model_performance_mini_{state.lower()}na_local_trends_200901--202312.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        dfm_mini_local_trends_09_19[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends/performance/Model_performance_mini_{state.lower()}na_local_trends_200901--201912.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        dfm_mini_local_trends_21_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends/performance/Model_performance_mini_{state.lower()}na_local_trends_202101--202312.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_trends_joint_09_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_joint/performance/Model_performance_mini_{state.lower()}na_local_trends_joint_200901--202312.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_trends_joint_09_19[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_joint/performance/Model_performance_mini_{state.lower()}na_local_trends_joint_200901--201912.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_trends_joint_21_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_joint/performance/Model_performance_mini_{state.lower()}na_local_trends_joint_202101--202312.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        dfm_mini_local_trends_all_09_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_all/performance/Model_performance_mini_{state.lower()}na_local_trends_all_200901--202312.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        dfm_mini_local_trends_all_09_19[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_all/performance/Model_performance_mini_{state.lower()}na_local_trends_all_200901--201912.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        dfm_mini_local_trends_all_21_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_all/performance/Model_performance_mini_{state.lower()}na_local_trends_all_202101--202312.xlsx',
                                        sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_trends_all_joint_09_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_all_joint/performance/Model_performance_mini_{state.lower()}na_local_trends_all_joint_200901--202312.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_trends_all_joint_09_19[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_all_joint/performance/Model_performance_mini_{state.lower()}na_local_trends_all_joint_200901--201912.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T
        # dfm_mini_local_trends_all_joint_21_23[state] = pd.read_excel(f'{perf_path}/mini_{state.lower()}na_local_trends_all_joint/performance/Model_performance_mini_{state.lower()}na_local_trends_all_joint_202101--202312.xlsx',
        #                                 sheet_name="performance", usecols="C:V",skiprows=6,nrows=4)[3:].T

                
    perf_dict = {'arma_09_23': arma_09_23, 'arma_09_19': arma_09_19, 'arma_21_23': arma_21_23,
                 'dfm_mini_local_09_23': dfm_mini_local_09_23, 'dfm_mini_local_09_19': dfm_mini_local_09_19, 'dfm_mini_local_21_23': dfm_mini_local_21_23,
                 #'dfm_mini_local_joint_09_23': dfm_mini_local_joint_09_23, 'dfm_mini_local_joint_09_19': dfm_mini_local_joint_09_19, 'dfm_mini_local_joint_21_23': dfm_mini_local_joint_21_23,
                 'dfm_mini_local_trends_09_23': dfm_mini_local_trends_09_23, 'dfm_mini_local_trends_09_19': dfm_mini_local_trends_09_19, 'dfm_mini_local_trends_21_23': dfm_mini_local_trends_21_23,
                 #'dfm_mini_local_trends_joint_09_23': dfm_mini_local_trends_joint_09_23, 'dfm_mini_local_trends_joint_09_19': dfm_mini_local_trends_joint_09_19, 'dfm_mini_local_trends_joint_21_23': dfm_mini_local_trends_joint_21_23,
                 'dfm_mini_local_trends_all_09_23': dfm_mini_local_trends_all_09_23, 'dfm_mini_local_trends_all_09_19': dfm_mini_local_trends_all_09_19, 'dfm_mini_local_trends_all_21_23': dfm_mini_local_trends_all_21_23,
                 #'dfm_mini_local_trends_all_joint_09_23': dfm_mini_local_trends_all_joint_09_23, 'dfm_mini_local_trends_all_joint_09_19': dfm_mini_local_trends_all_joint_09_19, 'dfm_mini_local_trends_all_joint_21_23': dfm_mini_local_trends_all_joint_21_23
                 }

    return perf_dict
                
def create_map_df(perf_dict, columns, states, offsets, time_periods):
    df = pd.DataFrame(columns=columns)
    df['state'] = states
    models = ['arma', 'dfm_mini_local', 'dfm_mini_local_joint', 'dfm_mini_local_trends', 'dfm_mini_local_trends_joint', 'dfm_mini_local_trends_all', 'dfm_mini_local_trends_all_joint']
    for model in models:
        for time_period in time_periods:
            for offset in offsets:
                df[f'{model}{time_period}_perf_{offset}'] = perf_dict[f'{model}{time_period}'].loc[int(offset)].tolist()
                df[f'mini_over_arma{time_period}_perf_{offset}'] = df[f'dfm_mini_local{time_period}_perf_{offset}'] - df[f'arma{time_period}_perf_{offset}']
                #df[f'joint_over_mini{time_period}_perf_{offset}'] = df[f'dfm_mini_local_joint{time_period}_perf_{offset}'] - df[f'dfm_mini_local{time_period}_perf_{offset}']
                df[f'trends_over_mini{time_period}_perf_{offset}'] = df[f'dfm_mini_local_trends{time_period}_perf_{offset}'] - df[f'dfm_mini_local{time_period}_perf_{offset}']
                #df[f'joint_trends_over_trends{time_period}_perf_{offset}'] = df[f'dfm_mini_local_trends_joint{time_period}_perf_{offset}'] - df[f'dfm_mini_local_trends{time_period}_perf_{offset}']
                df[f'all_over_trends{time_period}_perf_{offset}'] = df[f'dfm_mini_local_trends_all{time_period}_perf_{offset}'] - df[f'dfm_mini_local_trends{time_period}_perf_{offset}']
                #df[f'joint_trends_all_over_all{time_period}_perf_{offset}'] = df[f'dfm_mini_local_trends_all_joint{time_period}_perf_{offset}'] - df[f'dfm_mini_local_trends_all{time_period}_perf_{offset}']
    
    df.to_csv('state_performance_joint.csv')
    return df

def merge_geospatial(df):
    # wget.download("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip")
    gdf = gpd.read_file(os.getcwd()+'\cb_2023_us_state_500k')
    gdf = gdf.merge(df,left_on='STUSPS',right_on='state')

    return gdf

def plot_heatmaps(gdf, columns_to_plot, output_path):
    # set the value column that will be visualised
    for column in columns_to_plot:
        variable = column

        # make a column for value_determined_color in gdf
        # set the range for the choropleth values with the upper bound the rounded up maximum value
        vmin, vmax = gdf[variable].min(), gdf[variable].max()
        # Choose the continuous colorscale "YlOrBr" from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        if 'over' in variable:
            colormap = "RdYlGn"
        else:
            colormap = "YlGn"

        gdf, norm = makeColorColumn(gdf,variable,vmin,vmax)

        # create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
        visframe = gdf.to_crs({'init':'epsg:2163'})

        # create figure and axes for Matplotlib
        fig, ax = plt.subplots(1, figsize=(18, 14))
        # remove the axis box around the vis
        ax.axis('off')

        # set the font for the visualization to Helvetica
        hfont = {'fontname':'Helvetica'}

        # add a title and annotation
        # title = pretty_title(variable)
        # ax.set_title(f'{title}', **hfont, fontdict={'fontsize': '42', 'fontweight' : '1'})

        # Create colorbar legend
        fig = ax.get_figure()
        # add colorbar axes to the figure
        # This will take some iterating to get it where you want it [l,b,w,h] right
        # l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
        cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

        cbax.set_title('Out-of-Sample R-squared', **hfont, fontdict={'fontsize': '15', 'fontweight' : '0'})

        # add color scale
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # reformat tick labels on legend
        sm._A = []
        comma_fmt = FuncFormatter(lambda x, p: format(x, '.1%'))
        fig.colorbar(sm, cax=cbax, format=comma_fmt)
        tick_font_size = 16
        cbax.tick_params(labelsize=tick_font_size)
        # annotate the data source, date of access, and hyperlink
        # ax.annotate("Data: USDA Economic Research Service, accessed 15 Jan 23\nhttps://www.ers.usda.gov/topics/food-nutrition-assistance/food-security-in-the-u-s/key-statistics-graphics/#map", xy=(0.22, .085), xycoords='figure fraction', fontsize=14, color='#555555')

        # create map
        # Note: we're going state by state here because of unusual coloring behavior when trying to plot the entire dataframe using the "value_determined_color" column
        for row in visframe.itertuples():
            if row.state not in ['AK','HI']:
                vf = visframe[visframe.state==row.state]
                c = gdf[gdf.state==row.state][0:1].value_determined_color.item()
                vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

        # add Alaska
        akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])   
        akax.axis('off')
        # polygon to clip western islands
        polygon = Polygon([(-170,50),(-170,72),(-140, 72),(-140,50)])
        alaska_gdf = gdf[gdf.state=='AK']
        alaska_gdf.clip(polygon).plot(color=gdf[gdf.state=='AK'].value_determined_color, linewidth=0.8,ax=akax, edgecolor='0.8')

        # add Hawaii
        hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])   
        hiax.axis('off')
        # polygon to clip western islands
        hipolygon = Polygon([(-160,0),(-160,90),(-120,90),(-120,0)])
        hawaii_gdf = gdf[gdf.state=='HI']
        hawaii_gdf.clip(hipolygon).plot(column=variable, color=hawaii_gdf['value_determined_color'], linewidth=0.8,ax=hiax, edgecolor='0.8')

        print(f"Saving to: {output_path}/{variable}.png")
        print(f"Length: {len(output_path + '/' + variable + '.png')}")
        print(f"Variable repr: {repr(variable)}")

        fig.savefig(f'{output_path}/{variable}_main.png',dpi=400, bbox_inches="tight")
        #fig.savefig(f'{output_path}/{variable}.png',dpi=400, bbox_inches="tight")
        # bbox_inches="tight" keeps the vis from getting cut off at the edges in the saved png
        # dip is "dots per inch" and controls image quality.  Many scientific journals have specifications for this
        # https://stackoverflow.com/questions/16183462/saving-images-in-python-at-a-very-high-quality
        plt.close(fig)

def makeColorColumn(gdf,variable,vmin,vmax):
    # apply a function to a column to create a new column of assigned colors & return full frame
    if 'over' in variable:
        norm = mcolors.CenteredNorm(vcenter=0, clip=True)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlGn)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGn)
    gdf['value_determined_color'] = gdf[variable].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
    return gdf, norm

def pretty_title(variable):
    # Split string by '_'
    parts = variable.split('_')

    # Get offset at the end
    offset = parts[-1]

    # Get years before 'perf'
    end_index = parts.index('perf')
    years = parts[end_index - 2:end_index]  # start and end years
    years = ['20'+ y for y in years]

    # Get model type from the beginning up to the years
    model_parts = parts[:end_index - 2]
    model_raw = '_'.join(model_parts)

    # Pretty model names
    model_map = {
        'arma': 'ARMA(1,1)',
        'dfm_mini_local': 'DFM w/o Trends',
        'dfm_mini_local_trends': 'DFM w/ Local Trends',
        'dfm_mini_local_trends_all': 'DFM w/ Local and US Trends',
        'mini_over_arma': 'DFM w/o Trends - ARMA(1,1)',
        'trends_over_mini': 'DFM w/ Local Trends - DFM w/o Trends',
        'all_over_trends': 'DFM w/ Local and US Trends - DFM w/ Local Trends'
    }

    model = model_map.get(model_raw, model_raw.upper())

    # Format title
    return f"{model} — {years[0]}–{years[1]} — Offset {offset}"


if __name__ == '__main__':
    states = ['AK', 'AL', 'AR', 'AZ', 'CO', 'CT', 'DC', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY',
            'LA', 'MA', 'ME', 'MI', 'MO', 'MT', 'NC', 'NJ', 'NV', 'NY', 'OR', 'SC', 'TN', 'VA', 'WA', 'WI',
            'CA', 'MD', 'MN', 'NE', 'NH', 'NM', 'OH', 'OK', 'PA', 'TX', 'UT',
            'DE', 'MS', 'ND', 'SD', 'VT', 'WY', 'RI', 'WV',
            ]
    perf_dict = scrape_performance(states, 'L:/kyle/trends_nowcast/local_performance/eval_0')

    models = ['arma', 'dfm_mini_local', 'dfm_mini_local_trends', 'dfm_mini_local_trends_all', 'mini_over_arma', 'trends_over_mini', 'all_over_trends'
              #'dfm_mini_local_joint', 'dfm_mini_local_trends_joint', 'dfm_mini_local_trends_all_joint', 'joint_over_mini' 'joint_trends_over_trends', 'joint_trends_all_over_all'
              ]
    offsets = [str(20)]
    #offsets = [str(num) for num in range(-30,66,5)]
    time_periods = ['_09_23']
    #time_periods = ['_09_23', '_09_19', '_21_23']
    columns = ['state']
    for model in models:
        for time_period in time_periods:
            for offset in offsets:
                columns.append(model+time_period+'_perf_'+offset)
    
    df = create_map_df(perf_dict, columns, states, offsets, time_periods)
    gdf = merge_geospatial(df)

    columns_to_plot = []
    models_to_plot = ['arma', 'dfm_mini_local', 'dfm_mini_local_trends', 'dfm_mini_local_trends_all', 'mini_over_arma', 'trends_over_mini', 'all_over_trends']
    # #models_to_plot = ['mini_over_arma', 'trends_over_mini', 'all_over_trends']
    #offsets_to_plot = [str(num) for num in range(-30,66,5)]
    offsets_to_plot = [str(20)]
    #time_periods_to_plot = ['_09_23', '_09_19', '_21_23']
    time_periods_to_plot = ['_09_23']
    for model in models_to_plot:
        for time_period in time_periods_to_plot:
            for offset in offsets_to_plot:
                columns_to_plot.append(model+time_period+'_perf_'+offset)

    plot_heatmaps(gdf, columns_to_plot, 'C:/python_local_cd/Trends_Nowcasting/heatmaps/eval_0')
