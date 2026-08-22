# Attribution and project provenance

This project is built on top of the open-source Sign Language Translator project by Mudassar Iqbal and contributors.

- Original repository: https://github.com/sign-language-translator/sign-language-translator
- Original license: Apache License 2.0

## What comes from the original project

The core language-model logic used in this demo is based on the original project’s n-gram and mixer language-model components, which are reused here as part of the existing package API.

## What I added

The following pieces were added for this portfolio/demo work:

- the lightweight demo pipeline in [sign_language_translator/demo/pipeline.py](sign_language_translator/demo/pipeline.py)
- the heuristic sign classifier stub used by the demo
- the browser frontend and live demo UI
- the WebSocket-backed demo server and deployment configuration

## Required license notice

This project includes the original Apache-2.0 license from the upstream repository. Please keep the existing license file at the repository root intact.

Copyright 2023 Mudassar Iqbal <mdsriqb@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
