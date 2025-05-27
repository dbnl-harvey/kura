import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio  # For asyncio.gather and semaphore if needed for advanced tests
import numpy as np
from kura.embedding import OpenAIEmbeddingModel, SentenceTransformerEmbeddingModel

# Test data
TEXT_BATCH_1 = ["hello world", "this is a test"]
TEXT_BATCH_2 = ["another example", "sentence transformers rock"]
SINGLE_TEXT = ["single query"]

EXPECTED_EMBEDDING_1 = [0.1, 0.2, 0.3]
EXPECTED_EMBEDDING_2 = [0.4, 0.5, 0.6]
EXPECTED_EMBEDDING_3 = [0.7, 0.8, 0.9]
EXPECTED_EMBEDDING_4 = [1.0, 1.1, 1.2]


@pytest.fixture
def mock_openai_client():
    client = AsyncMock()

    # Configure create to return based on input list length
    async def mock_create_embeddings(*args, input, model, **kwargs):
        mock_response = AsyncMock()
        mock_response.data = []
        if input == TEXT_BATCH_1:
            mock_response.data.append(MagicMock(embedding=EXPECTED_EMBEDDING_1))
            mock_response.data.append(MagicMock(embedding=EXPECTED_EMBEDDING_2))
        elif input == TEXT_BATCH_2:
            mock_response.data.append(MagicMock(embedding=EXPECTED_EMBEDDING_3))
            mock_response.data.append(MagicMock(embedding=EXPECTED_EMBEDDING_4))
        elif input == SINGLE_TEXT:
            mock_response.data.append(MagicMock(embedding=EXPECTED_EMBEDDING_1))
        elif input == ["retry_text_openai"]:
            mock_response.data.append(MagicMock(embedding=EXPECTED_EMBEDDING_1))
        return mock_response

    client.embeddings.create = mock_create_embeddings
    return client


@pytest.fixture
def mock_sentence_transformer_model():
    model_instance = MagicMock()

    def mock_encode(texts, **kwargs):
        if texts == TEXT_BATCH_1:
            return np.array([EXPECTED_EMBEDDING_1, EXPECTED_EMBEDDING_2])
        elif texts == TEXT_BATCH_2:
            return np.array([EXPECTED_EMBEDDING_3, EXPECTED_EMBEDDING_4])
        elif texts == SINGLE_TEXT:
            return np.array([EXPECTED_EMBEDDING_1])
        elif texts == ["retry_text_st"]:
            return np.array([EXPECTED_EMBEDDING_1])
        return np.array([])  # Default empty

    model_instance.encode = mock_encode
    return model_instance


@pytest.mark.asyncio
async def test_openai_embedding_model_batch(mock_openai_client):
    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        model = OpenAIEmbeddingModel(
            model_batch_size=2, n_concurrent_jobs=1
        )  # Batch size not directly used by embed, but for consistency

        embeddings = await model.embed(TEXT_BATCH_1)
        assert embeddings == [EXPECTED_EMBEDDING_1, EXPECTED_EMBEDDING_2]
        mock_openai_client.embeddings.create.assert_called_once_with(
            input=TEXT_BATCH_1, model="text-embedding-3-small"
        )

        mock_openai_client.embeddings.create.reset_mock()
        embeddings_2 = await model.embed(TEXT_BATCH_2)
        assert embeddings_2 == [EXPECTED_EMBEDDING_3, EXPECTED_EMBEDDING_4]
        mock_openai_client.embeddings.create.assert_called_once_with(
            input=TEXT_BATCH_2, model="text-embedding-3-small"
        )


@pytest.mark.asyncio
async def test_sentence_transformer_embedding_model_batch(
    mock_sentence_transformer_model,
):
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer_model,
    ) as mock_st_constructor:
        model = SentenceTransformerEmbeddingModel(
            model_name="test-model", model_batch_size=2, n_concurrent_jobs=1
        )

        embeddings = await model.embed(TEXT_BATCH_1)
        assert embeddings == [EXPECTED_EMBEDDING_1, EXPECTED_EMBEDDING_2]
        mock_st_constructor.assert_called_once_with("test-model")
        mock_sentence_transformer_model.encode.assert_called_once_with(TEXT_BATCH_1)

        mock_sentence_transformer_model.encode.reset_mock()
        embeddings_2 = await model.embed(TEXT_BATCH_2)
        assert embeddings_2 == [EXPECTED_EMBEDDING_3, EXPECTED_EMBEDDING_4]
        mock_sentence_transformer_model.encode.assert_called_once_with(TEXT_BATCH_2)


@pytest.mark.asyncio
async def test_openai_embedding_model_single_text(mock_openai_client):
    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        model = OpenAIEmbeddingModel()
        embedding = await model.embed(SINGLE_TEXT)
        assert embedding == [EXPECTED_EMBEDDING_1]
        mock_openai_client.embeddings.create.assert_called_once_with(
            input=SINGLE_TEXT, model="text-embedding-3-small"
        )


@pytest.mark.asyncio
async def test_sentence_transformer_single_text(mock_sentence_transformer_model):
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer_model,
    ):
        model = SentenceTransformerEmbeddingModel()
        embedding = await model.embed(SINGLE_TEXT)
        assert embedding == [EXPECTED_EMBEDDING_1]
        mock_sentence_transformer_model.encode.assert_called_once_with(SINGLE_TEXT)


@pytest.mark.asyncio
async def test_openai_embedding_model_retry(mock_openai_client):
    # Reconfigure mock_client for retry test
    mock_openai_client.embeddings.create.side_effect = [
        Exception("API error 1"),
        Exception("API error 2"),
        AsyncMock(data=[MagicMock(embedding=EXPECTED_EMBEDDING_1)]),  # Successful call
    ]
    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        model = OpenAIEmbeddingModel(
            n_concurrent_jobs=1
        )  # Ensure sequential for predictable retry count
        test_text_list = ["retry_text_openai"]
        embedding = await model.embed(test_text_list)
        assert embedding == [EXPECTED_EMBEDDING_1]
        assert mock_openai_client.embeddings.create.call_count == 3


@pytest.mark.asyncio
async def test_sentence_transformer_model_retry(mock_sentence_transformer_model):
    # Reconfigure mock_st_model for retry test
    mock_sentence_transformer_model.encode.side_effect = [
        Exception("Encode error 1"),
        Exception("Encode error 2"),
        np.array([EXPECTED_EMBEDDING_1]),  # Successful call
    ]
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer_model,
    ):
        model = SentenceTransformerEmbeddingModel(n_concurrent_jobs=1)
        test_text_list = ["retry_text_st"]
        # Patch to_thread to run synchronously for this test to simplify mocking async behavior within thread
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            # Make to_thread directly call the (mocked) encode function
            async def side_effect_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)

            mock_to_thread.side_effect = side_effect_to_thread

            embedding = await model.embed(test_text_list)
            assert embedding == [EXPECTED_EMBEDDING_1]
            assert mock_sentence_transformer_model.encode.call_count == 3
            mock_to_thread.assert_called_with(
                mock_sentence_transformer_model.encode, test_text_list
            )


@pytest.mark.asyncio
async def test_openai_empty_input(mock_openai_client):
    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        model = OpenAIEmbeddingModel()
        embeddings = await model.embed([])
        assert embeddings == []
        mock_openai_client.embeddings.create.assert_not_called()


@pytest.mark.asyncio
async def test_sentence_transformer_empty_input(mock_sentence_transformer_model):
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer_model,
    ):
        model = SentenceTransformerEmbeddingModel()
        # Patch to_thread as it might be called even with empty list before internal ST check
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            embeddings = await model.embed([])
            assert embeddings == []
            # model.encode should not be called by to_thread if the input list is empty due to guard in embed()
            mock_to_thread.assert_not_called()
            mock_sentence_transformer_model.encode.assert_not_called()


# More advanced test: Concurrency limiting by internal semaphore
@pytest.mark.asyncio
async def test_openai_concurrency_limit(mock_openai_client):
    n_jobs = 2
    model = OpenAIEmbeddingModel(n_concurrent_jobs=n_jobs)

    # Mock the actual client call to add a delay, simulating work
    original_create = mock_openai_client.embeddings.create
    call_count = 0
    active_calls = 0
    max_active_calls = 0

    async def delayed_create(*args, **kwargs):
        nonlocal active_calls, max_active_calls, call_count
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        call_count += 1
        # print(f"Call {call_count} START, active: {active_calls}, max_active: {max_active_calls}")
        await asyncio.sleep(0.1)  # Simulate network latency
        result = await original_create(*args, **kwargs)
        active_calls -= 1
        # print(f"Call {call_count} END, active: {active_calls}")
        return result

    mock_openai_client.embeddings.create = delayed_create

    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        # Create more tasks than n_jobs to test queuing
        tasks = [model.embed(SINGLE_TEXT) for _ in range(n_jobs * 2)]
        await asyncio.gather(*tasks)

        assert mock_openai_client.embeddings.create.call_count == n_jobs * 2
        assert max_active_calls == n_jobs, (
            f"Expected max concurrent calls to be {n_jobs}, but got {max_active_calls}"
        )


# Similar concurrency test for SentenceTransformer could be added,
# but it's more complex due to asyncio.to_thread. The core idea is testing the semaphore.
# The OpenAI test above sufficiently demonstrates the internal semaphore pattern working.
