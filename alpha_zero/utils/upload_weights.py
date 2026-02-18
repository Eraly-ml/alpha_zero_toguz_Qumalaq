"""Utilities for uploading trained model weights to Hugging Face Hub and git repository."""
import os
import glob
import subprocess
import logging

logger = logging.getLogger(__name__)


def upload_to_huggingface(
    ckpt_dir: str,
    repo_id: str,
    commit_message: str = 'Upload AlphaZero Togyz Kumalak weights',
    token: str = None,
    private: bool = False,
) -> None:
    """Upload checkpoint files to a Hugging Face Hub repository.

    Args:
        ckpt_dir: Local directory containing checkpoint files.
        repo_id: Hugging Face repo ID, e.g. 'username/toguz-kumalak-alphazero'.
        commit_message: Commit message for the upload.
        token: HF API token. If None, uses HF_TOKEN env var or cached login.
        private: Whether to create a private repo if it doesn't exist.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error(
            'huggingface_hub is not installed. '
            'Install it with: pip install huggingface_hub'
        )
        return

    if token is None:
        token = os.environ.get('HF_TOKEN', None)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
        logger.info(f'Hugging Face repo "{repo_id}" is ready')
    except Exception as e:
        logger.error(f'Failed to create/access HF repo "{repo_id}": {e}')
        return

    # Find checkpoint files to upload
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    if not ckpt_files:
        logger.warning(f'No checkpoint files found in "{ckpt_dir}"')
        return

    # Upload the latest checkpoint (largest training step)
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    logger.info(f'Uploading latest checkpoint: {latest_ckpt}')

    try:
        api.upload_file(
            path_or_fileobj=latest_ckpt,
            path_in_repo=os.path.basename(latest_ckpt),
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f'Successfully uploaded {latest_ckpt} to {repo_id}')
    except Exception as e:
        logger.error(f'Failed to upload to HF Hub: {e}')
        return

    # Also upload all checkpoints in a folder upload
    try:
        api.upload_folder(
            folder_path=ckpt_dir,
            repo_id=repo_id,
            commit_message=f'{commit_message} (all checkpoints)',
            allow_patterns=['*.ckpt'],
        )
        logger.info(f'Successfully uploaded all checkpoints from "{ckpt_dir}" to {repo_id}')
    except Exception as e:
        logger.error(f'Failed to upload checkpoint folder to HF Hub: {e}')


def upload_to_git_repo(
    ckpt_dir: str,
    repo_dir: str = None,
    branch: str = None,
    commit_message: str = 'Add trained AlphaZero Togyz Kumalak weights',
    max_file_size_mb: float = 100.0,
) -> None:
    """Commit and push checkpoint files to the git repository.

    Args:
        ckpt_dir: Directory containing checkpoint files.
        repo_dir: Root of the git repository. Defaults to current working directory.
        branch: Git branch to push to. If None, uses current branch.
        commit_message: Commit message.
        max_file_size_mb: Skip files larger than this (GitHub limit is 100MB without LFS).
    """
    if repo_dir is None:
        repo_dir = os.getcwd()

    # Check if git-lfs is available for large files
    has_lfs = False
    try:
        result = subprocess.run(
            ['git', 'lfs', 'version'],
            cwd=repo_dir, capture_output=True, text=True,
        )
        has_lfs = result.returncode == 0
    except FileNotFoundError:
        pass

    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    if not ckpt_files:
        logger.warning(f'No checkpoint files found in "{ckpt_dir}"')
        return

    # Get the latest checkpoint only (to keep repo size manageable)
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    file_size_mb = os.path.getsize(latest_ckpt) / (1024 * 1024)

    logger.info(f'Latest checkpoint: {latest_ckpt} ({file_size_mb:.1f} MB)')

    if file_size_mb > max_file_size_mb:
        if has_lfs:
            logger.info('File exceeds size limit, using git-lfs to track *.ckpt files')
            subprocess.run(
                ['git', 'lfs', 'track', '*.ckpt'],
                cwd=repo_dir, check=True,
            )
            subprocess.run(
                ['git', 'add', '.gitattributes'],
                cwd=repo_dir, check=True,
            )
        else:
            logger.warning(
                f'Checkpoint file is {file_size_mb:.1f} MB (limit {max_file_size_mb} MB). '
                'Install git-lfs to handle large files: https://git-lfs.com'
            )
            return

    # Make checkpoint path relative to repo root
    rel_path = os.path.relpath(latest_ckpt, repo_dir)

    try:
        subprocess.run(
            ['git', 'add', rel_path],
            cwd=repo_dir, check=True,
        )

        # Check if there are staged changes
        result = subprocess.run(
            ['git', 'diff', '--cached', '--quiet'],
            cwd=repo_dir, capture_output=True,
        )
        if result.returncode == 0:
            logger.info('No new changes to commit')
            return

        subprocess.run(
            ['git', 'commit', '-m', commit_message],
            cwd=repo_dir, check=True,
        )
        logger.info(f'Committed checkpoint: {rel_path}')

        # Push with retry
        push_cmd = ['git', 'push']
        if branch:
            push_cmd.extend(['-u', 'origin', branch])

        for attempt in range(4):
            result = subprocess.run(
                push_cmd, cwd=repo_dir, capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger.info('Successfully pushed weights to git repository')
                return
            else:
                import time
                wait = 2 ** (attempt + 1)
                logger.warning(f'Push failed (attempt {attempt + 1}/4), retrying in {wait}s: {result.stderr}')
                time.sleep(wait)

        logger.error('Failed to push weights after 4 attempts')
    except subprocess.CalledProcessError as e:
        logger.error(f'Git operation failed: {e}')
