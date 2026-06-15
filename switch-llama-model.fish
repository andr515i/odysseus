#!/usr/bin/env fish

set -l model_name $argv[1]

if test -z "$model_name"
    echo "Usage: ./switch-llama-model.fish <gguf filename or substring>"
    echo
    echo "Available models:"
    find /home/quan/llama-stack/models -maxdepth 1 -type f -iname "*.gguf" -printf "%f\n" | sort
    exit 1
end

set -l matches (find /home/quan/llama-stack/models -maxdepth 1 -type f -iname "*$model_name*" -printf "%f\n" | sort)

if test (count $matches) -eq 0
    echo "No model matched: $model_name"
    exit 1
end

if test (count $matches) -gt 1
    echo "Multiple models matched:"
    printf "%s\n" $matches
    exit 1
end

set -l selected $matches[1]

perl -0pi -e "s|^LLAMA_MODEL=.*\$|LLAMA_MODEL=/models/$selected|m" .env

echo "Switched LLAMA_MODEL to:"
grep '^LLAMA_MODEL=' .env

docker compose up -d --force-recreate llama
