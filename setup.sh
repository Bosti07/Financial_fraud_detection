mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
maxUploadSize = 100\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
