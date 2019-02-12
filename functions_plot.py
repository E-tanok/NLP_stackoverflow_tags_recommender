def scatter_plot(Matrix,identifier_dataframe,cmap_categ,cmap_multiplier,title,size,screen_labels):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    """
    This function goal is to allow data visualization of 2D or 3D matrices of
    data with different attributes

    RESULT: A 2D or 3D visualization of the data, with different colors for
    each different categories. The centroids of each class are also ploted.
    (with the formula of unweighted barycenters)

    PARAMS :
    - 'Matrix' refers to a 2D or 3D Matrix of data
    - 'identifier_dataframe' must have 2 columns :
        --> "main_category" (String), the categorie's column of labels
        -->"color" (Integer), it's color transposition
    - 'cmap_categ' refers to the 'colormaps_reference' in matplotlib :
    https://matplotlib.org/examples/color/colormaps_reference.html
    - 'cmap_multiplier' is the multiplier to apply to the categories in order to
    scale the colors between 1 and 100
    - 'size' refers to the points sizes
    - 'screen_labels' refers to the way of displaying the point's categories:
        --> choose 'centroids' if you want to display the labels at the categories
        centroids levels
        --> let the string empty if you want to display one label each 50 points
    """

    if Matrix.shape[1] ==1:
        df_plot=pd.DataFrame(Matrix,columns =['X'])
        df_plot['Y']=1

    if Matrix.shape[1] ==2:
        df_plot=pd.DataFrame(Matrix,columns =['X','Y'])

    if Matrix.shape[1] ==3:
        df_plot=pd.DataFrame(Matrix,columns =['X','Y','Z'])

        fig = plt.figure(figsize=(13, 13))
        ax = plt.axes(projection='3d')

        min_X = min(df_plot.X)
        max_X = max(df_plot.X)

        min_Y = min(df_plot.Y)
        max_Y = max(df_plot.Y)

        min_Z = min(df_plot.Z)
        max_Z = max(df_plot.Z)


        # Data for a three-dimensional line
        xline = np.linspace(min_X, max_X, 50)
        yline = np.linspace(min_Y, max_Y, 50)
        zline = np.linspace(min_Z, max_Z, 50)
        ax.plot3D(xline, yline, zline, 'gray')

    new_identifier_df = pd.DataFrame(identifier_dataframe)
    new_identifier_df.index = range(0,len(new_identifier_df))

    df_plot['category']= new_identifier_df['main_category']
    df_plot['color']=new_identifier_df['color']


    if Matrix.shape[1] <=2:
        i=0
        dict_centroids = {}

        plt.figure(figsize=(13, 13))

        for categ in np.unique(list(df_plot['category'])):

            x = df_plot['X'][df_plot['category'] == categ]
            y = df_plot['Y'][df_plot['category'] == categ]

            #Calcul des centroïdes des catégories:
            Cx = x.mean()
            Cy = y.mean()

            L_centroids = [Cx,Cy]
            dict_centroids[categ] = L_centroids

            cmap = plt.get_cmap(cmap_categ)
            color = cmap(df_plot['color'][df_plot['category'] == categ]*cmap_multiplier)
            plt.scatter(x,y,s = size, c = color,label='%s'%categ)

        if screen_labels == 'centroids':
            # Affichage des labels des catégories au niveau des centroïdes :
            for categ in dict_centroids.keys():
                plt.scatter(dict_centroids[categ][0],dict_centroids[categ][1],c = 'black', s= 0.75*size)
                plt.text(dict_centroids[categ][0],dict_centroids[categ][1],categ, fontsize='14')

        else :
            #Affichage des labels des catégories (1 label sur 10) :
            X_ax = df_plot['X']
            Y_ax = df_plot['Y']
            for i, (x, y) in enumerate(zip(X_ax, Y_ax)):
                categ = df_plot['category'][i]
                if i%50 ==0:
                    plt.text(x, y,categ, fontsize='12')

        plt.title(title)
        plt.xlabel('dim_1')
        plt.ylabel('dim_2')
        plt.legend(loc='upper right')
        plt.show()

    if Matrix.shape[1] ==3:
        i=0
        dict_centroids = {}

        for categ in np.unique(list(df_plot['category'])):
            x = df_plot['X'][df_plot['category'] == categ]
            y = df_plot['Y'][df_plot['category'] == categ]
            z = df_plot['Z'][df_plot['category'] == categ]

            #Calcul des centroïdes des catégories:
            Cx = x.mean()
            Cy = y.mean()
            Cz = z.mean()

            L_centroids = [Cx,Cy,Cz]
            dict_centroids[categ] = L_centroids

            cmap = plt.get_cmap(cmap_categ)
            color = cmap(df_plot['color'][df_plot['category'] == categ]*cmap_multiplier)
            ax.scatter3D(x, y, z, c=color,label='%s'%categ)

        if screen_labels == 'centroids':
            # Affichage des labels des catégories au niveau des centroïdes :
            for categ in dict_centroids.keys():
                ax.scatter3D(dict_centroids[categ][0],dict_centroids[categ][1],dict_centroids[categ][2],c ='black', s= 0.75*size)
                ax.text(dict_centroids[categ][0],dict_centroids[categ][1],dict_centroids[categ][2],categ, fontsize='14')

        else :
            #Affichage des labels des catégories (1 label sur 50) :
            X_ax = df_plot['X']
            Y_ax = df_plot['Y']
            Z_ax = df_plot['Z']
            for i, (x, y, z) in enumerate(zip(X_ax, Y_ax,Z_ax)):
                categ = df_plot['category'][i]
                if i%50 ==0:
                    ax.text(x,y,z,categ, fontsize='12')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(title)
        fig.legend(loc='upper right')
        plt.show()

def radar_plot(myDataFrame,myIndex,myLabels,yticksList,yticks_str_List,lineColor,areaColor,figSizeTuple,title):
    from math import pi
    import numpy as np
    import matplotlib.pyplot as plt
    """
    This function goal is to visualize data with a radar plot

    RESULT : A radar plot

    PARAMS :
    - 'myDataFrame' refers to the entry DataFrame
    - 'myIndex' refers to the DataFrame index number used for the radar plot
    - 'yticksList' refers to the circulars iso distances presents in the radar plot
    - 'yticks_str_List' refers to the string labels associated to the 'yticksList'
    iso distances and show in the radar plot.
    - 'lineColor' refers to the trace borders colors.
    - 'areaColor' refers to the trace area color.
    - 'figSizeTuple' refers to a tuple specifiying the figure dimensions.
    """

    labels = myLabels
    stats = myDataFrame.loc[myIndex,myLabels].values

    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    # close the plot
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    fig= plt.figure(figsize=figSizeTuple)
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)

    ax.plot(angles, stats, 'o-', linewidth=2,color=lineColor)
    ax.fill(angles, stats, alpha=0.25,color=areaColor)
    plt.yticks(yticksList, yticks_str_List)
    plt.title(title)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.grid(True)
