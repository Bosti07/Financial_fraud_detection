mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
maxUploadSize = 110\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
