# Codestral Mamba for VSCode

An API which mocks [Llama.cpp](https://github.com/ggerganov/llama.cpp) to enable support for Codestral Mamba with the
[Continue Visual Studio Code extension](https://continue.dev/).

As of the time of writing and to my knowledge, this is the only way to use Codestral Mamba with VSCode locally. To make it work, we implement the `/completion` REST API from [Llama.cpp's HTTP server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md) and configure Continue for VSCode to use our server instead of Llama.cpp's. This way we handle all inference requests from Continue instead of Llama.cpp. When we get a request, we simply pass it off to [mistral-inference](https://github.com/mistralai/mistral-inference) which runs Continue's request with Codestral Mamba. Platform support is available wherever mistral-inference can be run.

Now let's get started!

### Setup

Prerequisites:
- [Download and run Codestral Mamba with mistral-inference](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) (Ref [1](https://colab.research.google.com/drive/1aHH4PW4eBU_R4R8pQ9BuYeOeMTiA98NF?usp=sharing#scrollTo=KWz9SwHXUfi-) & [2](https://github.com/mistralai/mistral-inference/releases/tag/v1.2.0) & [3](https://github.com/mistralai/mistral-inference/issues/192#issuecomment-2234242452))
- [Install the Continue VSCode extension](https://marketplace.visualstudio.com/items?itemName=Continue.continue)

After you are able to use both independently, we will glue them together with Codestral Mamba for VSCode.

Steps:
1. Install Flask to your mistral-inference environment with `pip install flask`.
2. Run `llamacpp_mock_api.py` with `python llamacpp_mock_api.py <path_to_codestral_folder_here>` under your mistral-inference environment.
3. Click the settings button at the bottom right of Continue's UI in VSCode and make changes to `config.json` so it looks like [this](https://docs.continue.dev/reference/Model%20Providers/llamacpp)[<sup>\[archive\]</sup>](http://web.archive.org/web/20240531162330/https://docs.continue.dev/reference/Model%20Providers/llamacpp). Replace `MODEL_NAME` with `mistral-8x7b`.

Restart VSCode or reload the Continue extension and you should now be able to use Codestral Mamba for VSCode!
