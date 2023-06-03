import solara


@solara.component
def Page():

    markdown = """
    ## Alpha Version - Alaskan Tundra 10-day Lightning Forecast

    Cloud to ground lightning 10-day lightning forecast for the Alaskan tundra.
    This is still work in progress and under development.

    ## Demo

    ![](https://drive.google.com/uc?export=view&id=1V9lpVfs13l4PYSlFmlZVCmeLlZXyw-zu)

    """

    solara.Markdown(markdown)
