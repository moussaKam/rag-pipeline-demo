import requests
from tqdm import tqdm
import wikipediaapi
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import pandas as pd
import argparse
from threading import Thread
import gradio as gr
from retriever import Retriever


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--title",
        type=str,
        default="الحرب العالمية الثانية",
        help="The title of the Wikipedia page to extract links from",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="The maximum number of workers to use for multithreading",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-awq",
        help="The name of the model to use for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to load the model onto",
    )
    parser.add_argument(
        "--wikipedia_language",
        type=str,
        default="ar",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-m3",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=3,
    )
    return parser.parse_args()


def generate_instruct_prompt(query, context, tokenizer):
    prompt = f"أجب على على السؤال التالي باللغة العربية وباستخدام السياق التالي: السياق: {context} \n\nالسؤال: {query}"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text


generation_kwargs = {
    "max_new_tokens": 600,
    "do_sample": False,
}


def extract_wikipedia_content(title, language_code="ar"):
    url = f"https://{language_code}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "prop": "extracts",
        "format": "json",
        "explaintext": True,
        "titles": "_".join(title.split()),
    }
    response = requests.get(url, params=params)
    data = response.json()
    # Extract the page content
    page = next(iter(data["query"]["pages"].values()))
    extract = page.get("extract", "No extract available")
    return extract


def extract_contents_multithreading(links, max_workers=5, language_code="ar"):
    contents = []

    # Using ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_link = {
            executor.submit(
                extract_wikipedia_content, link, language_code=language_code
            ): link
            for link in links
        }

        # Iterate over the completed futures as they finish
        for future in tqdm(as_completed(future_to_link), total=len(links)):
            link = future_to_link[future]
            try:
                contents.append(future.result())
            except Exception as exc:
                print(f"An error occurred while processing link {link}: {exc}")

    return contents


def get_wikipedia_links(page_title, language_code="ar"):
    # Set custom user-agent string
    user_agent = "MyWikipediaBot/1.0 (https://mywebsite.com/contact) MyBotName"

    wiki = wikipediaapi.Wikipedia(
        user_agent=user_agent,  # Specify user-agent to comply with Wikipedia's policy
        language=language_code,
    )

    page = wiki.page(page_title)

    if page.exists():
        return [str(el) for el in page.links]

    else:
        print("Page does not exist.")


def chunk_text(text, chunk_size=700, overlap=150):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks


def main(args):
    title = "_".join(args.title.split())
    links = get_wikipedia_links(title)
    contents = extract_contents_multithreading(links, args.max_workers)
    model_name = args.model_name
    device = args.device

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    chunks = [
        chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for text in contents
    ]
    chunks = [chunk for sublist in chunks for chunk in sublist]

    retriever = Retriever(args.embedding_model, device=device)

    retriever.encode_documents(chunks)

    def instruct(
        query,
    ):
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

        indices = retriever.retrieve([query], top_k=args.num_chunks)

        context = "\n\n".join([chunks[i] for i in indices])

        prompt = generate_instruct_prompt(query, context, tokenizer)

        print(prompt)

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        streamer = TextIteratorStreamer(
            tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs["input_ids"] = input_ids
        generation_kwargs["streamer"] = streamer

        t = Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()

        output_text = ""
        for new_text in streamer:
            output_text += new_text
            yield output_text, prompt
        return output_text, prompt

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        with gr.Row():
            query = gr.Textbox(
                lines=3,
                max_lines=8,
                interactive=True,
                label="query",
                rtl=True,
            )
        with gr.Row():
            answer = gr.Textbox(
                placeholder="",
                label="Answer",
                elem_id="q-input",
                lines=5,
                interactive=False,
                rtl=True,
            )
        with gr.Row():
            context = gr.Textbox(
                placeholder="",
                label="Answer",
                elem_id="q-input",
                lines=5,
                interactive=False,
                rtl=True,
                visible=False,
            )

        with gr.Row():
            submit = gr.Button("Submit")

        submit.click(
            instruct,
            inputs=query,
            outputs=[answer, context],
        )

    demo.launch(
        server_port=8085,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
