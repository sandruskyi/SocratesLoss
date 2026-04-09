"""
@User: sandruskyi
"""
import pandas as pd
import plotly.express as px

def risk_coverage_curve_plotly(risk, coverage, ds_type=None, path=None, loss="", paper = False, name_file ='', seed=''):
    coverage = coverage/100
    #print("risk plot", risk)
    #print("coverage plot", coverage)
    #print("Plotting risk-coverage curve")
    data = {"risk": risk, "coverage": coverage}
    df = pd.DataFrame(data)

    if paper:
        trace1 = px.line(df, x="coverage", y="risk", markers=True,
                         labels={
                             'risk': "Selective Risk (%)",
                             'coverage': 'Coverage'
                         },
                         #title="Risk-coverage curve. Es = " + seed.split('_')[0] +  ". Loss = " + " ".join(seed.split('_')[1:-1])
                         )
    else:
        trace1 = px.line(df, x="coverage", y="risk", markers=True,
                         labels={
                             'risk': "Selective Risk (%)",
                             'coverage': 'Coverage'
                         },
                         title="Risk-coverage curve"
                         )


    #trace1.update_xaxes(tickmode = 'array',tickvals = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    #trace1.update_yaxes(tickmode='array',tickvals=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #trace1.update_traces(texttemplate="%{text}",
    #                     textposition='top right')

    #trace1.add_annotation(
    #    text=("ECE = %.2f" % (ECE*100)), showarrow=False, xref="paper",
    #    yref='paper', x=0.97, y=0.07, xshift=-1, yshift=-5, font=dict(size=20, color="black"))
    #trace1.add_annotation(
    #    text=("MCE = %.2f" % (MCE*100)), showarrow=False, xref="paper",
    #    yref='paper', x=0.97, y=0.05, xshift=-1, yshift=-5, font=dict(size=20, color="black"))

    #trace1.update_layout(font=dict(size=20), legend={'traceorder': 'normal'})
    #trace1['data'][1]['line']['width'] = 4

    #trace1.show()
    if paper:
        trace1.update_layout(hovermode='x unified',
                             plot_bgcolor='white',
                             title_x=0.5,
                             font=dict(size=35, ),
                             # title={"y" : 0.88, "font":dict(size=25, )},
                             title={"font": dict(size=35, )},
                             xaxis=dict(mirror=True, showline=True, linecolor='black', linewidth=5, showgrid=True,
                                        gridcolor='rgba(0,0,0,0.1)'),
                             yaxis=dict(mirror=True, showline=True, linecolor='black', linewidth=5, showgrid=True,
                                        gridcolor='rgba(0,0,0,0.1)'),
                             legend_title=None)
        trace1.for_each_trace(lambda t: t.update(line=dict(width=5)) if t.line.dash == 'solid' else (
            t.update(line=dict(width=5, dash='dashdot'))))

        img_path = path + '/rc_' + seed + "_" + path.split("_")[1]  + "_" + path.split("_")[3] + '.png'
        trace1.write_image(img_path, width=1000, height=750)
        html_path = path + '/rc_' + seed  + "_" + path.split("_")[1]  + "_" + path.split("_")[3] + '.html'
        trace1.write_html(html_path)
    else:
        trace1.write_image(path + f"risk_coverage_curve_{ds_type}_{loss}.png",
                       width=1500, height=1000)
    print("Plotting finished")




