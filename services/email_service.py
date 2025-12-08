import requests
from os import environ as env
from services.logging_config import get_logger

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

class EmailService:
    def __init__(self):
        self.mailchimp_api_key = env.get("MAILCHIMP_API_KEY")
        self.mailchimp_list_id = env.get("MAILCHIMP_LIST_ID")  # This is the ID of your audience/list in Mailchimp
        self.mailchimp_base_url = "https://usX.api.mailchimp.com/3.0"  # Replace 'X' with the datacenter specific to your Mailchimp account

    def send_email_with_image(self, user_email, image_url, prompt, memory_content):
        """
        Sends an email to the user with the generated image, image prompt, and memory content.
        """
        # Create the email data
        email_data = {
            "email_address": user_email,
            "status": "subscribed",
            "merge_fields": {
                "IMAGE_URL": image_url,
                "PROMPT": prompt,
                "MEMORY_CONTENT": memory_content
            }
        }

        # Add the user to the list and send the email
        try:
            response = requests.post(
                f"{self.mailchimp_base_url}/lists/{self.mailchimp_list_id}/members",
                headers={"Authorization": f"Bearer {self.mailchimp_api_key}"},
                json=email_data
            )
            response.raise_for_status()

            # Note: Depending on your Mailchimp setup, you might need to trigger the actual sending of the email.
            # This code just adds/updates the user to the list with the provided merge fields.

        except requests.RequestException as e:
            logger.error(f"Error sending email via Mailchimp: {e}")
            return False

        return True

# Usage example:
# email_service = EmailService()
# email_service.send_email_with_image(user_email, image_url, prompt, memory_content)
