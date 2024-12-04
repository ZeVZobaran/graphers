# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 18:32:22 2021

Série de funções gráficas

@author: jzobaran
"""

# %% Setup, funções basicas, templates
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm


LAYOUT_BASE = dict(
            template='simple_white',
            title=dict(
                    font=dict(color='black', size=18)
                    ),
            font_family='Metropolis',
            yaxis=dict(
                    tickfont=dict(size=13),
                    color='black',
                    titlefont=dict(size=14),
                    linecolor='black',
                    tickprefix="<b>",
                    ticksuffix ="</b>",
                    tickformat=',.1f',
                    showgrid=True,
                    gridwidth=0.15,
                    gridcolor='gray',

                    title=dict(standoff=3)
                    ),
            yaxis2=dict(
                    tickfont=dict(size=13),
                    color='black',
                    titlefont=dict(size=14),
                    linecolor='black',
                    tickprefix="<b>",
                    ticksuffix ="</b>",
                    tickformat=',.1f',
                    title=dict(standoff=3)
                    ),
            legend=dict(
                    font=dict(color='black', size=13),
                    orientation='h'
                    ),
            xaxis=dict(
                    tickfont=dict(size=13),
                    color='black',
                    tickformat='<b>%b-%y</b>',
                    linecolor='black',
                    title=dict(standoff=3)
                    ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width=675,
            height=600,
            margin=dict(r=40, l=50, b=10, t=75)
            )

COLORSCALE = ['rgb(49,54,149)', 'rgb(224,224,224)', 'rgb(165,0,38)']

def COLORS():
    colors_generator =  (x for x in [
        'forestgreen', 'yellowgreen', 'dimgray', 'royalblue', 'firebrick',
        'chocolate', 'cadetblue', 'crimson', 'navy', 'goldenrod', 'lightgoldenrodyellow',
        'mediumblue', 'darkorange', 'red', 'deepskyblue', 'red'
            ])
    return colors_generator


def _names_preprocesser(df, series, nomes, eixo_x):
    if eixo_x:
        df_graf = df.set_index(eixo_x)
    else:
        df_graf = df.copy()
    df_graf = df_graf.loc[:, series]
    if nomes:
        renamer = {
            series[i]: f'<b>{nomes[i]}</b>' for i in range(len(series))
            }
        df_graf = df_graf.rename(renamer, axis=1)
    return df_graf

    


def gen_tabela(
        df_tabela, titulo,
        largura=2200, altura=375
        ):
    '''
    Tranforma um df de dados em uma tabela bonitinha no formato ibiuna
    '''
    fig = go.Figure(data=[go.Table(
        columnwidth=[800, 450],
        header=dict(
                values=[f'<b>{x}</b>' for x in df_tabela.columns],
                line_color='black',
                fill_color='rgba(55, 86, 35, 1)',
                align=['center', 'center'],
                font=dict(color='white', size=22),
                height=30
                ),
        cells=dict(
                values=[
                        list(df_tabela.loc[:, x].values)
                        for x in df_tabela.columns
                        ],  # Colunas
                line_color='black',
                fill_color=[
                        ['rgba(198, 224, 180, 1)', 'rgba(226, 239, 218, 1)']
                        * len(df_tabela),
                        ],
                align=['center', 'center'],
                font=dict(size=22, color='black'),
                height=40
                )
        )
    ])

    fig.update_layout(
            width=largura, height=altura,
            title=dict(
                    font_family='Metropolis',
                    text=f'<b>{titulo}</b>',
                    font=dict(
                            color='black',
                            size=30
                            ),
                    xanchor='left',
                    x=0.0,
                    yanchor='top'
                    ),
             margin=dict(l=10, r=10, b=10, t=50),
             plot_bgcolor='rgba(0,0,0,0)'
                )
    return fig


def gen_graf_dot_line(
        df, eixo_x, series_dict, titulo,
        unidade=1, cor='darkgreen', subtitulo=False
        ):
    '''
    gera gráficos de MM7D, com o dado diário pontilhado e alfado e o MM sólido
    Recebe df,eixos_x (coluna do df que dá o eixo x)
    series_dict (um dicionário label:coluna para as séries. Primeiro a diaria),
    titulo do gráfico, colorscheme: lista com as cores,
    subtitulo (um subtitulo não negrito opcional)
    '''
    fig = go.Figure()
    estilo = (x for x in ['dot', 'solid'])
    alfa = (x for x in [0.5, 1])
    for nome, coluna in series_dict.items():
        fig.add_trace(go.Scatter(
                x=df.loc[:, eixo_x],
                y=df.loc[:, coluna]/unidade,
                name=f'<b>{nome}<b>',
                line=dict(
                    color=cor, width=3, dash=next(estilo)
                    ),
                opacity=next(alfa)
                ))
    max_range = round(
        max(df.loc[:, series_dict.values()].max()*1.1)/unidade/10
        )*10
    fig.update_yaxes(range=[0, max_range])

    if subtitulo:
        fig.update_layout(
                LAYOUT_BASE,
                title=dict(
                        text=f'<b>{titulo}</b>'
                        f'<br>{subtitulo}'
                        )
                )
    else:
        fig.update_layout(
                LAYOUT_BASE,
                title=dict(
                        text=f'<b>{titulo}</b>'
                        )
                )
    return fig


def gen_graf_spread(
        df, titulo, serie_1, serie_2, nome_barras='spread',
        nomes=False, subtitulo=False, eixo_x=False
        ):
    '''
    Parameters
    ----------
    -------
    gráfico
    '''
    spread = df.loc[:, serie_1] - df.loc[:, serie_2]
    colors = COLORS()
    if eixo_x:
        df_graf = df.set_index(eixo_x)
    else:
        df_graf = df.copy()
    if nomes:
        df_graf = df_graf.rename({
            f'{serie_1}': nomes[0],
            f'{serie_2}': nomes[1]
            }, axis=1)
        serie_1 = nomes[0]
        serie_2 = nomes[1]
    df_graf = df_graf.loc[:, [serie_1, serie_2]]

    # Usamos esse df para gerar o gráfico, mas para manipulações usamos o df
    # já que o df não tem NaN
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
            x=df_graf.index,
            y=df_graf.loc[:, serie_1],
            name=f'<b>{serie_1}</b>',
            line=dict(color=next(colors), width=2)
            ),
        secondary_y=False
        )
    fig.add_trace(go.Scatter(
            x=df_graf.index,
            y=df_graf.loc[:, serie_2],
            name=f'<b>{serie_2}</b>',
            line=dict(color=next(colors), width=2)
            ),
        secondary_y=False
        )

    fig.add_trace(go.Bar(
            x=df_graf.index,
            y=spread,
            name=f'<b>{nome_barras} (e.d.)</b>',
            marker_color='black',
            opacity=0.5
            ),
            secondary_y=True
            )
    if subtitulo:
        fig.update_layout(
                LAYOUT_BASE,
                title=dict(
                        text=f'<b>{titulo}</b>'
                        f'<br>{subtitulo}'
                        )
                )
    else:
        fig.update_layout(
                LAYOUT_BASE,
                title=dict(
                        text=f'<b>{titulo}</b>'
                        )
                )
    # Outputter
    return fig



def gen_scatter(
        df, titulo, eixo_y=False, series=False, subtitulo=False, nomes=False,
        trendline=False, display_eq=False, display_r2=False,
        destacar_ultimo=False, colors=False, marker_line_width=1,
        legenda=False, eixo_x=False
        ):
    '''
    Parameters
    ----------
    df - Pandas Dataframe com as colunas que vão virar as linhas do graf
    titulo - Título do gráfico
    series - nomes das colunas que vão virar linhas
    nomes - opcional. Nomes de display das colunas. 
        Se False, usa os nomes das colunas mesmo
    eixos_x - opcional. Coluna a usar como eixo x. Se falso, usa o índice
    estilos - opcional. Mapper nomes: estilos para alterar estilos de linhas
    em particular. Se não especificado, linha contínua
    -------
    gráfico
    '''
# =============================================================================
#     #FIXME
#     # gambiarra do caralho, tomar cuidado
#     # estamos no processo de limpar. Conforme outros códigos sejam rodados
#     # e não funcionem, vamos arrumando. Se houver urgência, pode só descomentar
#     # esse snippet
#     if type(eixo_y) == bool and type(eixo_x) != bool:
#         eixo_y = eixo_x
# =============================================================================
    if not series:
        series = df.columns
    df_graf = _names_preprocesser(df, series, nomes, eixo_y)
    if not colors:
        colors = COLORS()
    # Usamos esse df para gerar o gráfico, mas para manipulações usamos o df
    # já que o df não tem NaN
    equacoes = ''  # Para adicionar as eqs OLSs
    fig = go.Figure()
    for coluna in df_graf.columns:
        cor = next(colors)
        y = df_graf.index
        x = df_graf.loc[:, coluna]
        if trendline:
            # Se pedir trendline, adicionamos ela com o statsmodels
            x_use = sm.add_constant(x.reset_index().iloc[:, 1:])
            if len(x_use.columns) < 2:
                # Se sm add constant não funcionar (sabe deus pq)
                # adicionamos a constante na porrada
                x_use.loc[:, 'const'] = 1
                x_use = x_use.loc[:, [x_use.columns[1], x_use.columns[0]]]
            fit_results = sm.OLS(
                y.values, x_use.values, missing='drop'
                ).fit()
            
            results_y = fit_results.predict()
            results_x = x[
                np.logical_not(np.logical_or(np.isnan(y), np.isnan(x)))
                ]
            sinal = ['-' if fit_results.params[1] < 0 else '+'][0]
            eq = f'<b>{coluna} = {round(fit_results.params[0], 2)} {sinal} '\
                f'{abs(round(fit_results.params[1], 2))} * X</b><br>'
            r2 = f'<b>R<sup>2</sup> = {round(100*fit_results.rsquared, 0)}%</b><br>'
            eq += r2
            equacoes += eq
            # Adicionando a linha de tendência ao gráfico
            fig.add_trace(go.Scatter(
                    x=results_x,
                    y=results_y,
                    name=f'<b>fit_{coluna}</b>',
                    showlegend=False,
                    line=dict(color=cor, width=1, dash='dash')
                    )
                )
        size = 7
        opacidade = 0.75
        if destacar_ultimo:
            cor = [cor for i in x][:-1]
            cor.append('firebrick')
            size = [7 for i in x][:-1]
            size.append(10)

        fig.add_trace(go.Scatter(
                x=x,
                y=y,
                name=f'<b>{coluna}</b>',
                mode='markers',
                marker=dict(
                    size=size,
                    color=cor
                    ),
                opacity=opacidade,
                marker_line_width=marker_line_width,
                marker_line_color='black',
                showlegend=legenda
                ),
            )
    # Add as equações
    if display_eq:
        fig.add_annotation(text=equacoes,
                           font=dict(color='black'),
                          xref="paper", yref="paper",
                          x=1.0, y=1.0, showarrow=False)
    elif display_r2:
        fig.add_annotation(text=r2,
                           font=dict(color='black'),
                          xref="paper", yref="paper",
                          x=1.0, y=1.0, showarrow=False)
        

    if subtitulo:
            string_titulo = f'<b>{titulo}</b><br>{subtitulo}'
    else:
            string_titulo = f'<b>{titulo}</b>'


    fig.update_layout(
            LAYOUT_BASE,
            title=dict(
                    text=string_titulo
                    )
            )
    fig.update_xaxes(
        linecolor='black',
        tickprefix="<b>",
        ticksuffix ="</b>",
        tickformat=',.1f'
        )

    # Outputter
    return fig


def gen_graf(
        df, titulo, series=None,
        subtitulo=False, nomes=False, eixo_x=False, estilos={}, colors=False
        ):
    '''
    Parameters
    ----------
    df - Pandas Dataframe com as colunas que vão virar as linhas do graf
    titulo - Título do gráfico
    series - nomes das colunas que vão virar linhas
    nomes - opcional. Nomes de display das colunas. 
        Se False, usa os nomes das colunas mesmo
    eixos_x - opcional. Coluna a usar como eixo x. Se falso, usa o índice
    estilos - opcional. Mapper nomes: estilos para alterar estilos de linhas
    em particular. Se não especificado, linha contínua
    colors - generator de cores. Usa o default se não passado
    -------
    gráfico
    '''
    if series is None:
        series = df.columns
    df_graf = _names_preprocesser(df, series, nomes, eixo_x)
    if not colors:
        colors = COLORS()
    # Usamos esse df para gerar o gráfico, mas para manipulações usamos o df
    # já que o df não tem NaN
    fig = go.Figure()
    for coluna in df_graf.columns:
        if coluna not in estilos.keys():
            estilos[coluna] = 'lines'
        fig.add_trace(go.Scatter(
                x=df_graf.index,
                y=df_graf.loc[:, coluna],
                name=f'<b>{coluna}</b>',
                line=dict(color=next(colors), width=2),
                mode=estilos[coluna]
                )
            )
    if subtitulo:
            string_titulo = f'<b>{titulo}</b><br>{subtitulo}'
    else:
            string_titulo = f'<b>{titulo}</b>'

    fig.update_layout(
            LAYOUT_BASE,
            title=dict(
                    text=string_titulo
                    )
            )
    if df_graf.min().min() < 0:
        fig.update_yaxes(
            zeroline=True, zerolinecolor='black'
            )

    # Outputter
    return fig

def gen_graf_barras(
        df, titulo, series, subtitulo=False, nomes=False, eixo_x=False
        ):
    '''
    idema à gen_graf, mas gera gráficos de barras
    '''
    df_graf = _names_preprocesser(df, series, nomes, eixo_x)
    df_graf = pd.DataFrame(df_graf)
    colors = COLORS()
    # Usamos esse df para gerar o gráfico, mas para manipulações usamos o df
    # já que o df não tem NaN
    fig = go.Figure()
    for coluna in df_graf.columns:
        fig.add_trace(go.Bar(
                x=df_graf.index,
                y=df_graf.loc[:, coluna],
                name=f'<b>{coluna}</b>',
                marker_color=next(colors)
                )
            )
    if subtitulo:
            string_titulo = f'<b>{titulo}</b><br>{subtitulo}'
    else:
            string_titulo = f'<b>{titulo}</b>'


    fig.update_layout(
            LAYOUT_BASE,
            title=dict(
                    text=string_titulo
                    )
            )
    # Outputter
    return fig


def gen_graf_dois_eixos(
        df, titulo, serie_1, serie_2, nomes=False, subtitulo=False,
        eixo_x=False
        ):
    '''
    Parameters
    ----------
    -------
    gráfico
    '''
    df_graf = df.copy()
    if nomes:
        df_graf = df_graf.rename({
            f'{serie_1}': nomes[0],
            f'{serie_2}': nomes[1]
            }, axis=1)
        serie_1 = nomes[0]
        serie_2 = nomes[1]
    df_graf = df_graf.loc[:, [serie_1, serie_2]]

    # Usamos esse df para gerar o gráfico, mas para manipulações usamos o df
    # já que o df não tem NaN
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
            x=df_graf.index,
            y=df_graf.loc[:, serie_1],
            name=f'<b>{serie_1}</b>',
            line=dict(color='forestgreen', width=2)
            ),
        secondary_y=False
        )
    fig.add_trace(go.Scatter(
            x=df_graf.index,
            y=df_graf.loc[:, serie_2],
            name=f'<b>{serie_2} (e.d.)</b>',
            line=dict(color='royalblue', width=2)
            ),
        secondary_y=True
        )

    if subtitulo:
        fig.update_layout(
                LAYOUT_BASE,
                title=dict(
                        text=f'<b>{titulo}</b>'
                        f'<br>{subtitulo}'
                        )
                )
    else:
        fig.update_layout(
                LAYOUT_BASE,
                title=dict(
                        text=f'<b>{titulo}</b>'
                        )
                )
    # Outputter
    return fig


def gen_graf_contrib(
        df, titulo, series, serie_total_nome='total',
        subtitulo=False, nomes=False, eixo_x=False
        ):
    '''
    Gera gráfico de barras empilhadas para representar uma contribuição, 
    com uma linha com markers para representar o total
    serie_total_nome deve ser o nome da série total, que será gerada a partir
    da soma das séries fornecidas
    '''
    df_graf = _names_preprocesser(df, series, nomes, eixo_x)
    df_graf.loc[:, serie_total_nome] = df_graf.sum(axis=1)

    colors = COLORS()
    # Usamos esse df para gerar o gráfico, mas para manipulações usamos o df
    # já que o df não tem NaN
    fig = go.Figure()
    for coluna in df_graf.columns:
        if coluna != serie_total_nome:
            fig.add_trace(go.Bar(
                    x=df_graf.index,
                    y=df_graf.loc[:, coluna],
                    name=f'<b>{coluna}</b>',
                    marker_color=next(colors)
                    )
                )
        else:
            fig.add_trace(go.Scatter(
                    x=df_graf.index,
                    y=df_graf.loc[:, coluna],
                    name=f'<b>{coluna}</b>',
                    line=dict(color='black', width=2),
                    mode='lines+markers'
                    )
                )

    if subtitulo:
            string_titulo = f'<b>{titulo}</b><br>{subtitulo}'
    else:
            string_titulo = f'<b>{titulo}</b>'

    # Add linha de zero
    fig.update_yaxes(zeroline=True, zerolinecolor='black')

    fig.update_layout(
            LAYOUT_BASE,
            title=dict(
                    text=string_titulo
                    ),
            barmode='relative'
            )
    min_xaxes = df_graf.index.min() - pd.DateOffset(months=3)
    max_xaxes = df_graf.index.max() + pd.DateOffset(months=3)

    fig.update_layout(xaxis_range=[min_xaxes, max_xaxes])
    # Outputter
    return fig




def gen_graf_curvas(
        df, titulo, vertices, nomes=False, linhas=[-30, -7, -2, -1], data_col='index'
        ):
    '''
    Gera gráfico das curvas de ativos a termo 
    em t0, t-linhas[0], t-linhas[1] e t-linhas[2]
    Linhas = índices a incluir no gráfico
    recebe um df com colunas com os vertices (vertices em ordem)
    vertices_nomes podem ser os nomes de cada vértice, tb em ordem
    se data_col = index, infere que as datas são o índice
    '''

    colors = (x for x in [
        'yellowgreen', 'darkseagreen', 'darkgreen', 'firebrick'
            ])
    estilos = (x for x in ['lines', 'lines', 'lines', 'lines+markers'])
    # Impõe os índice
    if data_col=='index':
        pass
    else:
        df = df.set_index(data_col)

    df_graf = df.iloc[linhas]
    df_graf = df_graf.loc[:, vertices].T

    if not nomes:  # Para poder negritar o eixo X sempre
        nomes = vertices

    renamer = {
        vertices[i]: f'<b>{nomes[i]}</b>' for i in range(len(vertices))
        }
    df_graf = df_graf.rename(renamer)

    fig = go.Figure()
    for data in df_graf.columns:
        try:
            nome = data.strftime('<b>%d-%m-%y</b>')
        except AttributeError:
            # Se não for um datetime, só negrita mesmo
            nome = f'<b>{data}</b>'
        fig.add_trace(go.Scatter(
                x=df_graf.index,
                y=df_graf[data],
                name=nome,
                line=dict(color=next(colors), width=2),
                mode=next(estilos)
                )
            )
    fig.update_layout(
            LAYOUT_BASE,
            title=dict(
                    text=f'<b>{titulo}</b>'
                    )
            )
    # Outputter
    return fig

def gen_graf_multiseries(
        df, titulo, series, subtitulo=False, series_principais=False,
        eixo_x=False, nomes=False,
        cor_principal=False,
        cor_secundaria='gray', gradiente=True, legenda=True,
        range_def=True
        ):
    '''
    Gera um gráfico com multiplas linhas pouco diferenciadas e uma opcional
    em destaque
    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe com as colunas a serem usadas no gráfico
            O índice será o eixo x, e os nomes, os columnames
    titulo : String
        Título do gráfico
    subtitulo : String, optional
        Subtitulo do gráfico
    series : List
        Nomes das colunas a usar como linhas parecidas no graf
    series_principais : List, optional
        Se True, nomes das colunas a usar como séries em destaque.
        The default is False.
    series_principais : List, optional
        Se True, nomes a usar para as séries em destaque.
        The default is False.
    cor_principal : Generator, optional
        Generator de cores para a principal. Default para 
        Preto - Vermelho. Para mais de duas, passar (Não use esse estilo tho)
    cor_secundaria : String, optional
        Cor das linhas "menos importantes". The default is 'gray'.
    gradiente : BOOL, optional
        Se as linhas "Menos importantes" devem fade. The default is True.
    legenda: BOOL, optional
        Se as linhas "Menos importantes" devem ter legenda. The default is True.

    Returns
    -------
    Graf - Gráfico desejado

    '''

    alfa = (
        x for x in np.linspace(0.2, 1, len(series))
        )
    if not cor_principal:
        cor_principal=(x for x in ['black', 'firebrick'])
    fig = go.Figure()
    for serie in series:
        fig.add_trace(go.Scatter(
        x=df.index,
        y=df[serie],
        name=f'<b>{serie}</b>',
        showlegend=legenda,
        line=dict(color=cor_secundaria, width=2),
        opacity=[next(alfa) if gradiente else 1][0]
        ))
    if series_principais:
        if not nomes:
            nomes = series_principais
        for i in range(len(series_principais)):
            serie = series_principais[i]
            nome = nomes[i]
            fig.add_trace(go.Scatter(
            x=df.index,
            y=df[serie],
            name=f'<b>{nome}</b>',
            line=dict(color=next(cor_principal), width=3),
            ))
    if subtitulo:
            string_titulo = f'<b>{titulo}</b><br>{subtitulo}'
    else:
            string_titulo = f'<b>{titulo}</b>'

    fig.update_layout(
            LAYOUT_BASE,
            title=dict(
                    text=string_titulo
                    )
            )
    if range_def:
        max_range = round(
            df.loc[:, series].max().median() + df.loc[:, series].std().max()*2
            )
        fig.update_yaxes(range=[0, max_range])

    # Outputter
    return fig


def gen_heatmap(
        df, titulo, series=None, subtitulo=False, nomes=False, transpose=False,
        texto=None, colorscale=COLORSCALE, eixo_no_topo=False, by='all',
        midpoint=None
        ):
    '''
    
    Parameters
    ----------
    df : pandas df
        dataframe a ser usado. Índice já deve estar selecionado!
    titulo : str
        título do gráfico
    series : listlike, optional
        Colunas do df a usar no gráfico. Default None --> usa todas as colunas
    subtitulo : str, optional
        subtitulo do gráfico. The default is False --> sem subtitulo.
    nomes : listlike, optional
        Se passar, usa como os nomes das series. Deve estar na ordem certa.
        The default is False.
    transpose : bool, optional
        Se o heatmap deve ser transposto. The default is False.
    texto : arraylike, optional
        Displays para os tiles do heatmap, já formatados. The default is None.
    colorscale : colorscale-like, optional
        colorscale para usar. The default is COLORSCALE (definida acima).
    eixo_no_topo: bool, optional
        se o eixo_x deve ser displayado no topo do gráfico. Default False
    by: one of 'all', 'columns', 'rows'
        como colorir o gráfico. Default all

    Returns
    -------
    fig : plotly chart
        o heatmap

    '''
    if series is None:
        series = df.columns

    df_graf = _names_preprocesser(df, series, nomes, eixo_x=False)

    if by == 'rows':
        # Normalizamos o df por coluna para as cores serem por col
        # O texto não muda!
        df_graf = ((df_graf.T - df_graf.mean(axis=1))/df_graf.std(axis=1)).T
    elif by == 'columns':
        df_graf = ((df_graf - df_graf.mean(axis=0))/df_graf.std(axis=0))
    else:
        pass


    fig = go.Figure(data=go.Heatmap(
        z=df_graf,
        x=df_graf.columns,
        y=df_graf.index,
        showscale=False,
        colorscale=colorscale,
        zmid=midpoint
        ))

    if texto is not None:
        fig.update_traces(
            text=texto,
            texttemplate='<b>%{text}</b>',
            textfont={'size': 14, 'family': 'Metropolis'}
            )

    if eixo_no_topo:
        fig.update_xaxes(
            ticks='',
            side='top'
            )
        fig.update_yaxes(
            autorange="reversed"
            )


    fig.update_xaxes(
            tickprefix = '<b>',
            ticksuffix = '</b>'
            )

    if subtitulo:
            string_titulo = f'<b>{titulo}</b><br>{subtitulo}'
    else:
            string_titulo = f'<b>{titulo}</b>'

    fig.update_layout(
            LAYOUT_BASE,
            title=dict(
                    text=string_titulo
                    ),
            yaxis=dict(showgrid=False)
            )
    fig.update_layout(plot_bgcolor='rgb(224,224,224)')

    return fig


# TODO converter em linhas
def gen_cometas(
        df, titulo, subtitulo, cometas, cometas_ref, opacidade_ref, eixo_x, eixo_y,
        eixo_x_nome, eixo_y_nome
        ):

    # converter as datas em uma coluna que relaciona as opacidades
    df.loc[:, 'opacity'] = (df[opacidade_ref] - df[opacidade_ref].min() + 0.1)/(
            df[opacidade_ref].max()-df[opacidade_ref].min()+0.1
            )
    opacidades = list(set(df.loc[:, 'opacity']))
    opacidades.sort()

    fig = go.Figure()
    for opacidade in opacidades:
        if opacidade == 1:
            legenda = True
        else:
            legenda = False
            # Só mostra a legenda do último, em que a cor será o mais
            # visivel possível
            
        colors = COLORS()
        for cometa in cometas:
            cor = next(colors)
            df_use = df.loc[df['opacity']==opacidade]
            df_use = df_use.loc[df_use[cometas_ref]==cometa]
            fig.add_trace(go.Scatter(
                    x=df_use[eixo_x],
                    y=df_use[eixo_y],
                    name=f'<b>{cometa}</b>',
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=cor
                        ),
                    opacity=opacidade,
                    marker_line_width=0,
                    marker_line_color=cor,
                    showlegend=legenda
                    )
                )
    if subtitulo:
            string_titulo = f'<b>{titulo}</b><br>{subtitulo}'
    else:
            string_titulo = f'<b>{titulo}</b>'


    fig.update_layout(
            LAYOUT_BASE,
            title=dict(
                    text=string_titulo
                    )
            )
    fig.update_xaxes(
        linecolor='black',
        tickprefix="<b>",
        ticksuffix ="</b>",
        tickformat=',.1f', 
        title=f'<b>{eixo_x_nome}</b>',
        zeroline=True
        )
    fig.update_yaxes(
        title=f'<b>{eixo_y_nome}</b>',
        zeroline=True, zerolinecolor='black'
        )
    return fig