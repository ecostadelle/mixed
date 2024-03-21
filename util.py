import os
import re
import sys

import pandas as pd

def get_filename():
    return re.split(r'[\\/\.]', sys.argv[0])[-2]

def get_pretty_filename():
    return get_filename().replace('_', ' ').title()

def create_table(index):
    multi = pd.MultiIndex.from_tuples([
        ('','Dataset'),
        ('', 'ID'),
        ('', 'OOD'),
        ('Strategy', '1'),
        ('Strategy', '2'),
        ('Strategy', '3'),
        ('Strategy', '4')])
    
    return pd.DataFrame(index=index, columns=multi)

def insert_line(multi_line_text: str, line_text:str, at_line: int):
    lines = multi_line_text.split('\n')
    lines.insert(at_line, line_text)
    return '\n'.join(lines)

def apply_style(df):
    cols = df.columns
    df_styled = df.style.hide_index()
    df_styled = df_styled.highlight_max(subset=cols[2:],axis=1,props='font-weight: bold')
    return df_styled.format("{:.03f}",subset=cols[1:], na_rep="")

def get_latex(df, caption):
    df_styled = apply_style(df)
    return df_styled.to_latex(
        convert_css=True,
        hrules=True,
        column_format='lcc|cccc', 
        multicol_align='c',
        caption=caption,
        position='h!',
        position_float="centering")

def summarize_strategies(df, number_of_strategies=4):
    return [(df.iloc[:,2]<df.iloc[:,3+i]).sum() for i in range(number_of_strategies)]

def personalize_latex(latex, summary):
    add_summary = '\\multicolumn{3}{l|}{Times the strategy outperformed the OOD} & ' 
    add_summary += ' & '.join(map(str, summary)) 
    add_summary += ' \\\\'
    
    latex = insert_line(latex, '\\begin{threeparttable}', 2)
    latex = insert_line(latex, '\\midrule', -4)
    latex = insert_line(latex, add_summary, -4)
    latex = insert_line(latex, '\\end{threeparttable}', -2)
    return insert_line(latex, '\n\\vspace{10pt}\n', -1)

def save_result(df_result, filename=get_filename(), 
                pretty_filename=get_pretty_filename(),
                caption=None):
    
    result = ''
    for table in df_result.keys():
        df = df_result[table]
        
        if not caption:
            caption = table.upper()
            caption += " for Model Adaptation with balanced ensemble of Categorical \\textit{Na\\\"ive} Bayes"
            caption += f' ({pretty_filename})'
        
        latex = get_latex(df, caption=caption)
        summary = summarize_strategies(df)
        result += personalize_latex(latex, summary)
    try: 
        os.makedirs('results', exist_ok=True)
        f = open(file=f'results/{filename}.tex', mode='w')
        f.write(result)
    except Exception as e:
        print(e)
