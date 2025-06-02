import pandas as pd
from tabulate import tabulate # type: ignore
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import os

def display_console_table(hyperparams:dict, filename='table.png', header:list=["Parameter", "Value"]):

    table_data = list(hyperparams.items())#, columns=["Parameter", "Value"]

    # df = pd.DataFrame(table_data)

    # Generate Markdown table with LaTeX
    markdown_table = tabulate(table_data, headers=header, tablefmt='github')

    # Display the table
    display(Markdown(markdown_table))

    # print(tabulate(df, headers='keys', tablefmt='grid'))

    """
    Directly dump the latex table:
    # Generate LaTeX table code
    latex_table = tabulate(table_data, headers=['Letter', 'Region Description'], tablefmt='latex')

    # Print the LaTeX table code
    print(latex_table)
    """

    # # Create a matplotlib figure and axis
    # fig, ax = plt.subplots(figsize=(8, len(df) * 0.5))  # Adjust figsize as needed
    
    # # Hide axes
    # ax.axis('tight')
    # ax.axis('off')
    
    # # Create the table
    # table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')
    
    # # Style the table (optional)
    # table.auto_set_font_size(False)
    # table.set_fontsize(12)
    # table.auto_set_column_width(col=list(range(len(df.columns))))
    
    # # Save the table as a PNG image
    # plt.savefig(filename, bbox_inches='tight', dpi=300)
    # plt.close()

    # print(f"Table saved as {filename}")

def generate_latex_table(hyperparams:dict, filename:str, save_dir:str , header:list=["Parameter", "Value"]):

    header_ = [rf'\textbf{{{h}}}' for h in header]
    tt = filename.rsplit('.', 1)[0]
    caption = tt+'.'

    if "Radio" in filename:
        to_add = r" Where $\bar \xi(0) =  \frac{1}{K} \sum\limits_{i \in \mathcal{K}} f\left(\xi_i(0)\right)$." #$\frac{1}{|\Phi_{BS}|} \sum\limits_{\substack{i \in \Phi_{BS}, \\ j \in i_E}} \xi_i^{\textnormal{sen}}(0) + \xi_{i,j}^{\textnormal{com}}(0)$
        caption +=to_add
    elif "Association" in filename:
        to_add = r" Where $F_X(x) =  \mathbb{P}(X \leq x)$ and $\rho_{X,Y}(x,y) = \frac{F_{X,Y}(x,y)}{F_X(x) \ F_Y(y)}$."
        caption +=to_add
    
    df = pd.DataFrame(list(hyperparams.items()), columns=header_)#columns=["Parameter", "Value"]
    latex_table = df.to_latex(index=False, 
                              caption=caption, 
                              label=f"tab:{tt}",
                              column_format='|c|c|',
                              header=True,
                              float_format="%.3f",
                              escape=False,
                              position='H')
    latex_table = latex_table.replace(r"\begin{table}[H]", r"\begin{table}[H]\centering", 1)
    if filename:
        # add \usepackage{{booktabs}}
        #simulator_output
        path = f"{save_dir}/params"
        os.makedirs(path, exist_ok=True)

        filename = f'{path}/{filename}'
        with open(filename, 'w') as f:
            f.write(latex_table)
    else:
        print(latex_table)

def generate_log(hyperparams:dict, title:str, save_dir=str, header:list=["Parameter", "Value"]):
    #hyperparams = get_hyperparameters()
    
    # Display in console
    display_console_table(hyperparams=hyperparams, filename=f'{title}.png', header=header)
    
    # Generate LaTeX and save to file
    generate_latex_table(hyperparams=hyperparams, filename=f"{title}.tex", header=header, save_dir=save_dir)

# if __name__ == "__main__":
#     generate_log()
