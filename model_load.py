import gdown

# a checkpoint
id_model = "1hBYrwcw96KIjjkBcH2GZawOWXAZ5lEXO"
output_model = "model_best.pt"
gdown.download(id=id_model, output=output_model, quiet=False)


# embedder model

id_embedder = '1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL'
output_embedder = "embedder.pt"
gdown.download(id=id_embedder, output=output_embedder, quiet=False)
