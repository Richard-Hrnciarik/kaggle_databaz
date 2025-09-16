# Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('API_ILO_country_YU.csv')
print(df.shape)
print(df.columns[:10])
print(df.head())

non_country_list=['Arab World','Central Europe and the Baltics','Caribbean small states','East Asia & Pacific (excluding high income)',
                 'Early-demographic dividend', 'East Asia & Pacific','Europe & Central Asia (excluding high income)',
                 'Europe & Central Asia','Euro area','European Union','Fragile and conflict affected situations','High income',
                 'Heavily indebted poor countries (HIPC)','IBRD only', 'IDA & IBRD total', 'IDA total','IDA blend','IDA only',
                 'Latin America & Caribbean (excluding high income)','Latin America & Caribbean','Least developed countries: UN classification', 
                 'Low income','Lower middle income', 'Low & middle income','Late-demographic dividend','Middle East & North Africa',
                 'Middle income','Middle East & North Africa (excluding high income)','North America','OECD members','Other small states',
                 'Pre-demographic dividend','Post-demographic dividend','South Asia','Sub-Saharan Africa (excluding high income)',
                 'Sub-Saharan Africa','Small states','East Asia & Pacific (IDA & IBRD countries)',
                 'Europe & Central Asia (IDA & IBRD countries)','Latin America & the Caribbean (IDA & IBRD countries)',
                 'Middle East & North Africa (IDA & IBRD countries)','South Asia (IDA & IBRD)',
                 'Sub-Saharan Africa (IDA & IBRD countries)','Upper middle income','World']

df_country = df[~df['Country Name'].isin(non_country_list)].copy()
# ponecháme len vybrané roky
years = ['2010','2011','2012','2013','2014']
cols = ['Country Name','Country Code'] + years
df_country = df_country[cols]
print(df_country.shape)
df_country.head()


df_non_country=df[df['Country Name'].isin(non_country_list)]

df_non_country.head()

df_non_country.shape

index=df_non_country.index

x_data = ['2010', '2011','2012', '2013','2014']

y0 = df_country['2010']
y1 = df_country['2011']
y2 = df_country['2012']
y3 = df_country['2013']
y4 = df_country['2014']

y_data = [y0,y1,y2,y3,y4]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)']

traces = []

# stĺpce s rokmi prevedieme na čísla (niekedy bývajú stringy)

for y in years:

    df_country[y] = pd.to_numeric(df_country[y], errors='coerce')

# zmena medzi rokmi (percentuálna hodnota v p.b.)
df_country['change_2012_2010'] = df_country['2012'] - df_country['2010']
df_country['change_2014_2012'] = df_country['2014'] - df_country['2012']
df_country['change_2014_2010'] = df_country['2014'] - df_country['2010']

df_country[['Country Name','change_2014_2010']].head()

for xd, yd, color in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            whiskerwidth=0.2,
            fillcolor=color,
            marker=dict(
                size=2,
            ),
            boxmean=True,    
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Distribution of Unemployment Data',
    xaxis=dict(
        title='Year'
    ),
    yaxis=dict(
        title='Unemployment Rate (%)',
        autorange=True,
        showgrid=True,
        zeroline=False,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
#        zerolinecolor='rgb(255, 255, 255)',
#        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)

l=[]
trace0= go.Scatter(
        y= df_country['2010'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2010'].values,
                    line= dict(width=1),
                    color= df_country['2010'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l.append(trace0);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2010',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False,
)
fig= go.Figure(data=l, layout=layout)

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale

l1=[]
trace1= go.Scatter(
        y= df_country['2011'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2011'].values,
                    line= dict(width=1),
                    color= df_country['2011'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l1.append(trace1);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2011',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l1, layout=layout)

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale

l2=[]
trace2= go.Scatter(
        y= df_country['2012'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2012'].values,
                    line= dict(width=1),
                    color= df_country['2012'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l2.append(trace2);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2012',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l2, layout=layout)

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale

l3=[]
trace3= go.Scatter(
        y= df_country['2013'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2013'].values,
                    line= dict(width=1),
                    color= df_country['2013'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l3.append(trace3);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2013',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l3, layout=layout)

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale

l4=[]
trace4= go.Scatter(
        y= df_country['2014'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2014'].values,
                    line= dict(width=1),
                    color= df_country['2014'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l4.append(trace4);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2014',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l4, layout=layout)

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale

df_country['2014-2012 change']=df_country['2014']-df_country['2012']

df_country['2012-2010 change']=df_country['2012']-df_country['2010']

# Tried with Plotly now going with seaborn
twoyearchange201412_bar, countries_bar1 = (list(x) for x in zip(*sorted(zip(df_country['2014-2012 change'], df_country['Country Name']), 
                                                             reverse = True)))

twoyearchange201210_bar, countries_bar2 = (list(x) for x in zip(*sorted(zip(df_country['2012-2010 change'], df_country['Country Name']), 
                                                             reverse = True)))

# Another direct way of sorting according to values is creating distinct sorted dataframes as in below commented ways and then
# passing their values directly as in below mentioned code to achieve the same effect as by above mentioned method.

# df_country_sorted=df_country.sort(columns='2014-2012 change',ascending=False)
# df_country_sorted.head()

top10 = df_country.sort_values('change_2014_2010').head(10)       # najväčší POKLES (lepšie)
bottom10 = df_country.sort_values('change_2014_2010', ascending=False).head(10)  # najväčší NÁRAST (horšie)

plt.figure(figsize=(10,6))
sns.barplot(x='change_2014_2010', y='Country Name', data=top10, orient='h')
plt.title('TOP 10 – najväčší pokles nezamestnanosti (2010→2014)')
plt.xlabel('Zmena v percentuálnych bodoch')
plt.ylabel('Krajina')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x='change_2014_2010', y='Country Name', data=bottom10, orient='h')
plt.title('BOTTOM 10 – najväčší nárast nezamestnanosti (2010→2014)')
plt.xlabel('Zmena v percentuálnych bodoch')
plt.ylabel('Krajina')
plt.tight_layout()
plt.show()

# Plotting 2010 World Unemployment Data Geographically
data = [ dict(
        type = 'choropleth',
        locations = df_country['Country Code'],
        z = df_country['2010'],
        text = df_country['Country Name'],
        colorscale = 'Reds',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Unemployment (%)'),
      ) ]

layout = dict(
    title = 'Unemployment around the globe in 2010',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean = True,
        #oceancolor = 'rgb(0,255,255)',
        oceancolor = 'rgb(222,243,246)',
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
#py.iplot( fig, validate=False)

# Colorscale Sets the colorscale and only has an effect if `marker.color` is set to a numerical array. 
# Alternatively, `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, 
# Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis

# Plotting 2014 World Unemployment Data Geographically
data = [ dict(
        type = 'choropleth',
        locations = df_country['Country Code'],
        z = df_country['2014'],
        text = df_country['Country Name'],
        colorscale = 'Reds',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Unemployment (%)'),
      ) ]

layout = dict(
    title = 'Unemployment around the globe in 2014',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean = True,
        #oceancolor = 'rgb(0,255,255)',
        oceancolor = 'rgb(222,243,246)',
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )

# Colorscale Sets the colorscale and only has an effect if `marker.color` is set to a numerical array. 
# Alternatively, `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, 
# Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis

# Plotting 2014 World Unemployment Data Geographically

choose = ['Slovak Republic','Czech Republic','Germany']  # uprav podľa dostupných názvov
sel = df_country[df_country['Country Name'].isin(choose)][['Country Name']+years].set_index('Country Name').T

plt.figure(figsize=(8,5))
for c in sel.columns:
    plt.plot(sel.index, sel[c], marker='o', label=c)
plt.title('Nezamestnanosť mladých (2010–2014) – vybrané krajiny')
plt.xlabel('Rok')
plt.ylabel('Nezamestnanosť (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()