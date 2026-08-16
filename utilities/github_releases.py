import requests


def get_latest_github_release(owner, repo):
    """
    Get the latest release from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    try:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            releases = response.json()
            if releases:
                return releases[0]  # Return the newest release dictionary
        return None
    except Exception:
        return None