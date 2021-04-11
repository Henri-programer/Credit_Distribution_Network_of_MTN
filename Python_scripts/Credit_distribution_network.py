######################################################
# Project: Credit Distribution Network of MTN-Benin  #
# Author:  Henri Noel Kengne, Junior Data Scientist  #
# Completion Date : January 3, 2020                  #
# nkengne@africanschoolofeconomics.com               #
######################################################


# THE NECESSARY MODULES WE WILL NEED
import networkx as nx
import pandas as pd
import plotly
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import os
import numpy as np

# for Notebook
# %matplotlib inline

# Organizing my Working directory
main_directory = 'C:/Users/PT WORLD/Desktop/My Projects/Network_project_MTN/'
os.chdir(main_directory)

newpath = r'./figures'  # store figures here
if not os.path.exists(newpath):
    os.makedirs(newpath)

# set seed for reproducibility
np.random.seed(1984) 


# PART 1: HERE WE INTENT TO USE A SMALL DATASET TO EXPLAIN HOW A SIMPLE VISUALIZATION OF 
# A CREDIT TERANSFER NETWORK CAN HELP IDENTIFY FRAUDULENT TRANSACTIONS. THE DATA USED HERE WAS SLIGHTLY
# MODIFIED FOR THIS PURPOSE.

# First, we are defining a simple method to draw the graph and the centrality metrics of nodes with a heat map.
def draw_graph(G, pos, measures, measure_name,path_save):
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),                   #measures.values(),
                                   nodelist= measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    
    #labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    #plt.axis('off')
    plt.savefig(path_save)
    plt.show()

# Read in the first data.
df = pd.read_csv("./Data/trydata.csv") # Our dataset is turn into a pandas dataframe
G = nx.DiGraph()   # Initialize the network as a directed network


# We can now build the weighted edges of the network 
for i,elrow in df.iterrows():
    G.add_edge(elrow[0],elrow[1], attr_dict = elrow[4:].to_dict(), weight = elrow['Amount'])

# The color of the edges depends on the sender's type.     
color_map = []
for node in G.nodes():
    for fa  in df.FromAccountID :
        if node == fa:
            if df.ers_sender_rs_type[df.loc[df.FromAccountID  == fa].index[0]]=='SC':
                color_map.append('red')
        
            if df.ers_sender_rs_type[df.loc[df.FromAccountID  == fa].index[0]]=='SCC': 
                color_map.append('saddlebrown')
            
            if df.ers_sender_rs_type[df.loc[df.FromAccountID  == fa].index[0]]=='MD': 
                color_map.append('black')
            #if df.ers_receiver_rs_type[df.loc[df.ToAccountID  == fa].index[0]]=='MMD':
                #color_map.append('orange')
            if df.ers_sender_rs_type[df.loc[df.FromAccountID  == fa].index[0]]=='SD':
                color_map.append('blue')
            
            if df.ers_sender_rs_type[df.loc[df.FromAccountID  == fa].index[0]]=='POS':
                color_map.append('green')
                
# We can manualy enter the cordinates of the different nodes since we are working with a small dataset.               
xpos = [0,-50,100,150,0,250,300,350,600,-60,0,50,120,300,460,550,700,850,1000]
ypos = [200,150,150,175,100,100,100,100,100,50,50,50,50,50,50,50,50,50,50]  

# Now that we have assigned  colors to the edges, we add an additional column that contains colors
#  to our dataframe
color = pd.DataFrame(data = color_map)
df['color'] = color

# We rebuild the edges of our network and specify their weight. 
for i,elrow in df.iterrows():
    G.add_edge(elrow[0],elrow[1], attr_dict = elrow[4:].to_dict(), weight = elrow['Amount'])

# Let's see what our dataframe looks like    
df.head()

# We create a dictionary that contains the nodes and their positions
node_col = [
'red','black','black','saddlebrown','blue','blue','blue','blue','blue','purple',
'purple','purple','purple','purple','purple','purple','purple','purple','purple'
]
nodeList = {'NodeName': G.nodes(), 'X':xpos, 'Y':ypos, 'C':node_col}
nodeFrame = pd.DataFrame(data=nodeList)

# Add node properties
for i,nlrow in nodeFrame.iterrows():
    G.node[nlrow[0]].update(nlrow[1:].to_dict()) 
    
# Let's plot the graph
node_pos = {node[0]: (node[1]['X'],node[1]['Y']) for node in G.nodes(data = True)}
edge_col = [e[2]['attr_dict']['color'] for e in G.edges(data = True)]
node_col = [
'red','black','black','saddlebrown','blue','blue','blue','blue','blue','purple','purple',
'purple','purple','purple','purple','purple','purple','purple','purple'
]

legend_elements = [
Line2D([0], [0],marker = 'o', color = 'red',label = 'SC (Service Centers)', markersize = 15),
Line2D([0], [0],marker = 'o', color = 'saddlebrown',label = 'SCC (Sub-service Centers)', markersize = 15),
Line2D([0], [0],marker = 'o', color = 'black',label = 'MD (Master Distributors)', markersize = 15),
Line2D([0], [0],marker = 'o', color = 'blue',label = 'SD (Sub-master Distributors)', markersize = 15),
Line2D([0], [0],marker = 'o', color = 'purple',label = 'POS (Points of Sales)', markersize = 15)
]

# Create the figure
fig,ax = plt.subplots(figsize = (20, 10))
ax.legend(handles = legend_elements)   
nx.draw_networkx(
G, pos = node_pos, arrows=True, edge_color = edge_col, node_size = 1500, 
alpha = .85, node_color = node_col,with_labels = True
)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(
G, pos = node_pos, edge_labels = labels,font_color = 'k', alpha = 2,font_weight = 'bold'
) # edge_labels=labels,
#plt.title('', size=15)
#plt.axis("off") # Uncomment this line to remove axes
plt.savefig('./figures/small_credit_transfer_network.png', dpi=300,bbox_inches='tight')
plt.show

# Visualizing the credit transactions: Hiveplot, Cercoplot and Arc plot.

# # Hiveplot.
from hiveplot import HivePlot
nodes = dict()
nodes['red'] = [n for n,d in G.nodes(data=True) if d['C'] == 'red']
nodes['saddlebrown'] = [n for n,d in G.nodes(data=True) if d['C'] == 'saddlebrown']
nodes['black'] = [n for n,d in G.nodes(data=True) if d['C'] == 'black']
nodes['blue'] = [n for n,d in G.nodes(data=True) if d['C'] == 'blue']
nodes['purple'] = [n for n,d in G.nodes(data=True) if d['C'] == 'purple']
edges = dict()
edges['group1'] = G.edges(data=True)
nodes_cmap = dict()
nodes_cmap['blue'] = 'blue'
nodes_cmap['red'] = 'red'
nodes_cmap['black'] = 'black'
nodes_cmap['purple'] = 'purple'
nodes_cmap['saddlebrown'] = 'saddlebrown'
edges_cmap = dict()
edges_cmap['group1'] = 'darkgreen'
h = HivePlot(nodes, edges, nodes_cmap, edges_cmap)
h.draw()
plt.savefig('./figures/hiveplot_credit_transaction.png',dpi=300,bbox_inches='tight')



import nxviz as nv # this library is use for cercoplot and arcplot

## MatrixPlot
A = nx.to_numpy_matrix(G) # Convert G to a matrix format: A
G_conv = nx.from_numpy_matrix(A, create_using = nx.DiGraph()) # Convert A back to the NetworkX form 
                                                            # as a directed graph: T_conv
Mat = nv.MatrixPlot(G_conv) # node_order='c', node_color='C',edge_color=None) 
                             # This creates a Matplot object Mat.
Mat.draw()             
#plt.show()             
plt.savefig('./figures/Matrixplot_credit_transaction.png')
x = nx.adj_matrix(G_conv)
print(x.todense())

## Cercoplot
from nxviz import CircosPlot
Mat = CircosPlot(G , node_labels=True,node_order='Y',font_size=32, node_color='C',
figsize=(8,8),font_weight='bold') # This creates a circosplot object Mat.
Mat.draw()  
plt.savefig('./figures/Cerco_credit_transaction.png')
plt.show()             

## Arcplot
from nxviz import ArcPlot
Mat = ArcPlot(G ,node_labels=True,font_size=25,node_order='Y', node_color='C',figsize=(8,8))
Mat.draw() 
plt.savefig('./figures/Arc_credit_transaction.png')           
plt.show() 
 
# Centrality Measures

## Degree Centrality
pos = node_pos
draw_graph(G, pos, nx.in_degree_centrality(G), 'In-degree Centrality','./figures/indegree_credit_transaction.png')
draw_graph(G, pos, nx.out_degree_centrality(G), 'Out-degree Centrality','./figures/outdegree_credit_transaction.png')

## Closeness Centrality
#draw_graph(G, pos, nx.closeness_centrality(G, distance='weight'), 'Closeness Centrality')
draw_graph(G, pos, nx.closeness_centrality(G, distance=None, wf_improved=True) ,'closeness Centrality',
'./figures/closeness_credit_transaction.png')



# PART 2: IN THIS PART WE WILL BE WORKING WITH REAL-WORLD DATA. WE WILL VIZUALIZE THE CREDIT MOVEMEMTS OF
# OF MTN-BENIN FOR THE SOLE PURPOSE OF IDENTIFYING SUSPICIOUS LINKS.


import missingno as mn  

# Read in the data
df = pd.read_csv("./Data/CDRtestdata.csv")
df.head() # Just to have a look at the first observations of the data frame
mn.matrix(df) # Visualize how much missing variables are in each columns?

Graph=nx.DiGraph()
for i,elrow in df.iterrows():
    Graph.add_edge(elrow[0],elrow[1],attr_dict=elrow[0:].to_dict()) 
    
# Here our program assigns color to nodes base on their type   
node_col = []
NodeSet = list(Graph.nodes()) 
for node in NodeSet:
    if (node in list(df.ers_from_partner_id) and df.ers_sender_rs_type[df.loc[df.ers_from_partner_id  == node].index[0]]=='SC') or (node in list(df.ers_to_partner_id) and df.ers_receiver_rs_type[df.loc[df.ers_to_partner_id  == node].index[0]]=='SC'):
        node_col.append('red')
        
    if (node in list(df.ers_from_partner_id) and df.ers_sender_rs_type[df.loc[df.ers_from_partner_id  == node].index[0]]=='SCC') or (node in list(df.ers_to_partner_id) and df.ers_receiver_rs_type[df.loc[df.ers_to_partner_id  == node].index[0]]=='SCC'):
        node_col.append('saddlebrown')
            
    if (node in list(df.ers_from_partner_id) and df.ers_sender_rs_type[df.loc[df.ers_from_partner_id  == node].index[0]]=='MD') or (node in list(df.ers_to_partner_id) and df.ers_receiver_rs_type[df.loc[df.ers_to_partner_id  == node].index[0]]=='MD'): 
        node_col.append('black')
        
    if (node in list(df.ers_from_partner_id) and df.ers_sender_rs_type[df.loc[df.ers_from_partner_id  == node].index[0]]=='MDD') or (node in list(df.ers_to_partner_id) and df.ers_receiver_rs_type[df.loc[df.ers_to_partner_id  == node].index[0]]=='MDD'):
        node_col.append('hotpink')
        
    if (node in list(df.ers_from_partner_id) and df.ers_sender_rs_type[df.loc[df.ers_from_partner_id  == node].index[0]]=='SD') or (node in list(df.ers_to_partner_id) and df.ers_receiver_rs_type[df.loc[df.ers_to_partner_id  == node].index[0]]=='SD'):
        node_col.append('blue')
            
    if (node in list(df.ers_from_partner_id) and df.ers_sender_rs_type[df.loc[df.ers_from_partner_id  == node].index[0]]=='MTN') or (node in list(df.ers_to_partner_id) and df.ers_receiver_rs_type[df.loc[df.ers_to_partner_id  == node].index[0]]=='MTN'):
        node_col.append('darkgreen')
    
    if (node in list(df.ers_from_partner_id) and df.ers_sender_rs_type[df.loc[df.ers_from_partner_id  == node].index[0]]=='POS') or (node in list(df.ers_to_partner_id) and df.ers_receiver_rs_type[df.loc[df.ers_to_partner_id  == node].index[0]]=='POS'):
        node_col.append('yellow')

# Here the program assigns colors to edges or links base on the type of the node they depart from.       
        
edge_col = [0 for i in range(len(NodeSet))]
for start in NodeSet:
    for i in list(df.loc[df.ers_from_partner_id == start].index):
        if df.ers_sender_rs_type[i] == 'SC':
            edge_col[i] ='red'
        if df.ers_sender_rs_type[i] == 'MD':
            edge_col[i] ='black'
        if df.ers_sender_rs_type[i] == 'MDD':
            edge_col[i] ='hotpink'
        if df.ers_sender_rs_type[i] == 'SD':
            edge_col[i] ='blue'
        if df.ers_sender_rs_type[i] == 'MTN':
            edge_col[i] ='darkgreen'
        if df.ers_sender_rs_type[i] == 'POS': # uncomment this line if there are POS in your dataset.
            edge_col[i] ='yellow'

import random

# We can assign random positions to the nodes
xpos = [random.randrange(0,7000) for i in range(len(NodeSet))]
ypos = [random.randrange(0,7000) for i in range(len(NodeSet))]

# The Graph edges and theirs weight can be built.
color= pd.DataFrame(data=edge_col)
df['color']=color
df.columns = [col.strip() for col in df.columns]
for i,elrow in df.iterrows():
    Graph.add_edge(elrow[0],elrow[1], attr_dict = elrow[2:].to_dict(), weight = elrow['Amount_transfered'])
nodeList = {'NodeName': Graph.nodes(), 'X':xpos, 'Y':ypos, 'C':node_col}
nodeFrame = pd.DataFrame(data=nodeList)

# add node properties
for i,nlrow in nodeFrame.iterrows():
    Graph.node[nlrow[0]].update(nlrow[1:].to_dict()) 

# plot the network
node_pos = {node[0]: (node[1]['X'],node[1]['Y']) for node in Graph.nodes(data=True)}
edge_col = [e[2]['attr_dict']['color'] for e in Graph.edges(data=True)]

legend_elements = [
Line2D([0], [0], marker ='o', color ='red', label='SC (Service Centers)', markersize=15),
Line2D([0], [0], marker ='o', color ='black', label='MD (Master Distributors)', markersize=15),
Line2D([0], [0], marker ='o', color ='hotpink', label='MDD (Master Distributor Duplicates)', markersize=15),
Line2D([0], [0], marker ='o', color ='darkgreen', label='MTN (Mobile Telephone Network)', markersize=15),
Line2D([0], [0], marker ='o', color ='blue', label='SD (Sub-Master Distributors)', markersize=15),
Line2D([0], [0], marker ='o', color ='yellow', label='POS (points of sale)', markersize=15)
]
                   
# Create the figure
fig , ax = plt.subplots(figsize=(25,25))
ax.legend(handles = legend_elements)   
nx.draw_networkx(Graph, 
pos = node_pos, arrows = True, edge_color = edge_col
, node_size = 1500, alpha = .85, node_color = node_col,with_labels = True
)
labels = nx.get_edge_attributes(Graph, 'weight')
nx.draw_networkx_edge_labels(Graph, pos=node_pos,edge_labels=labels,font_color='k', alpha=1)
plt.title('CDR Network', size=15)
#plt.axis("off") # Uncomment this line to remove the axis
plt.savefig('./figures/CDR_network.png', dpi=300,bbox_inches='tight')
plt.show

# Now let's look at the  Hiveplot, Cercoplot and Arc plot.

## Hiveplot 
from hiveplot import HivePlot
nodes = dict()
nodes['red'] = [n for n,d in Graph.nodes(data=True) if d['C'] == 'red']
nodes['darkgreen'] = [n for n,d in Graph.nodes(data=True) if d['C'] == 'darkgreen']
nodes['black'] = [n for n,d in Graph.nodes(data=True) if d['C'] == 'black']
nodes['blue'] = [n for n,d in Graph.nodes(data=True) if d['C'] == 'blue']
nodes['yellow'] = [n for n,d in Graph.nodes(data=True) if d['C'] == 'yellow']
nodes['hotpink'] = [n for n,d in Graph.nodes(data=True) if d['C'] == 'hotpink']
edges = dict()
edges['group1'] = Graph.edges(data=True)

nodes_cmap = dict()
nodes_cmap['blue'] = 'blue'
nodes_cmap['darkgreen'] = 'darkgreen'
nodes_cmap['red'] = 'red'
nodes_cmap['black'] = 'black'
nodes_cmap['yellow'] = 'yellow'
nodes_cmap['hotpink'] = 'hotpink'
edges_cmap = dict()
edges_cmap['group1'] = 'black'
h = HivePlot(nodes, edges, nodes_cmap, edges_cmap)
# plt.figure(figsize=(10, 10))
h.draw()
plt.savefig('./figures/CDR_hiveplot.png')

## Cercoplot
from nxviz import CircosPlot
Mat = CircosPlot(Graph , node_labels = True,font_size = 12,node_order = 'C', node_color ='C',figsize = (10,10)) 
Mat.draw()  
plt.savefig('./figures/CDR_circoplot.png')           
plt.show()             

## Arcplot
from nxviz import ArcPlot
Mat = ArcPlot(Graph,node_labels=True,font_size=12,node_order='C', node_color='C',figsize=(10,10)) 
Mat.draw() 
plt.savefig('./figures/CDR_arcplot.png')            
plt.show() 

# Centrality Measures

## Degree Centrality
pos = node_pos
draw_graph(Graph, pos, nx.in_degree_centrality(Graph), 'In-degree Centrality','./figures/indegree_CDR.png')
draw_graph(Graph, pos, nx.out_degree_centrality(Graph), 'Out-degree Centrality','./figures/outdegree_CDR.png')

## Closeness Centrality
#draw_graph(G, pos, nx.closeness_centrality(G, distance='weight'), 'Closeness Centrality')
draw_graph(Graph, pos, nx.closeness_centrality(Graph, distance=None, wf_improved=True) ,'closeness Centrality',
'./figures/closeness_CDR.png')


# PART 3: LET US BUILD A 3D CREDITS DISTRIBUTION NETWORK. 


import random
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# To have a similar graph in 3D, we will keep previous value of 2D position and add a third one
Xn = xpos #or Xn = [random.randrange(0,7000) for i in range(len(NodeSet))]
Yn = ypos #or Yn = [random.randrange(0,7000) for i in range(len(NodeSet))]
Zn = [random.randrange(0,7000) for i in range(len(NodeSet))] 
    
nodeList = {'NodeName': Graph.nodes(), 'X':Xn, 'Y':Yn, 'Z':Zn}
nodeFrame = pd.DataFrame(data=nodeList)

# add node properties
for i,nlrow in nodeFrame.iterrows():
    Graph.node[nlrow[0]].update(nlrow[1:].to_dict())  

amount = []
for i in labels:
    amount.append(labels[i])
    
group = edge_col
position = {node[0]: (node[1]['X'],node[1]['Y'],node[1]['Z']) for node in Graph.nodes(data=True)}
name = [x for x in Graph.nodes()]

# edges
trace1=go.Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=3), #width represent the weight of the line
               text=amount,
               hoverinfo='text'
               )
# nodes
trace2=go.Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=17,
                             color=group,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text=name,
               hoverinfo='text'
               )

axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout = go.Layout(
         title="Visualization of Credits Distribution Network of MTN-Benin (3D visualization)",
         width=1000,
         height=1000,
         showlegend=True,
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text="Important: click on trace0 to visualize edges or actors to visualize nodes",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ],    )


data=[trace1, trace2]
fig=go.Figure(data=data, layout=layout)

iplot(fig, filename='./figures/CDRs Network')   # visualize inside the notebook
plot(fig, filename='./figures/CDRs Network.html')    # visualize outside the notebook

