# Chat Bot using Gemini model

import pyodbc
import sqlalchemy
from sqlalchemy import create_engine,text
from configparser import ConfigParser
import google.generativeai as genai
from dotenv import load_dotenv
import os,re,json

load_dotenv()
system_instruction = os.environ.get("system_instruction")

generation_config = {
  "temperature": 0.5,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 2048,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]

class GenAIExeption(Exception):
    """GenAI Exception base class"""

class ChatBot:
    CHATBOT_NAME = 'AI Assistant'

    def __init__(self,api_key):
        self.genai = genai
        self.genai.configure(
            api_key=api_key
        )
        self.model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
        self.conversation = None
        self.dbflag = False 
        self._conversation_history = []
        self.preload_conversation()
        self.db_engine = self.setup_db_connection()

    # Getting Response as SQL Quary
    def get_gemini_response(self,question,conversation_history):
        response=self.model.generate_content([question])  
        response.resolve() 
        return response.text

    

    def _generation_config(self,temperature):
        return genai.types.GenerationConfig(
            temperature=temperature
        )
    
    
    
    def setup_db_connection(self):
        # config = ConfigParser()
        # config.read('creadential.ini')
        db_username = os.environ.get("db_username")
        db_password = os.environ.get("db_password")
        db_host = os.environ.get("db_hostname")
        db_name = os.environ.get("db_database")
        db_server = os.environ.get("db_server")

        connection_string = f"mssql+pyodbc://{db_username}:{db_password}@{db_server}/{db_name}?driver=ODBC Driver 17 for SQL Server"
        engine = create_engine(connection_string)
        return engine
    
    def handle_response(self, response):
        # Blocking potentially dangerous operations
        if any(keyword in response.upper() for keyword in ["UPDATE", "DELETE", "DROP", "TRUNCATE","CREATE"]):
            return "This operation is Not Possible."
        
        # Process non-SELECT responses
        if not response.strip().upper().startswith("SELECT") :
            response_parts = response.split('\n', 1)
            if len(response_parts) > 1:
                response = response_parts[1].split('`')[0].strip()
        
        # Execute the query if it starts with SELECT
        if response.strip().upper().startswith("SELECT"):
            print(f"Executing query: {response}")
            db_results = self.execute_queries(response)
            # product_attribute = response.split('SELECT ')[1].split(' FROM')[0]
            # print(product_attribute)
            
            if not db_results:
                db_results="No related products found."
        else:
            db_results = None

        return db_results
    
    def sanitize_input(self,user_input):
        # Only allow alphanumeric characters and spaces
        sanitized_input = re.sub(r'[^a-zA-Z0-9\s/-]', '', user_input)
        return sanitized_input

        

    
    
    
    def send_prompts(self, user_input, temperature=0.5):
        if(temperature<0 or temperature >1):
            raise GenAIExeption('Temperature must be between 0 and 1')
        if not user_input:
            raise GenAIExeption('Prompt can not be empty')
        
        # Sanitize and validate user input 
        user_input = self.sanitize_input(user_input)
        
        # self.dbflag = choice == 1
        
        # if self.dbflag:
        # Check if the prompt should bypass the database
        bypass_db_queries = ["hi", "hello", "hey", "no", "yes", "bye"]
        if user_input.lower() not in bypass_db_queries:
            try:
                response = self.get_gemini_response(user_input,conversation_history=self._conversation_history)
                self._conversation_history.append({"role": "user", "content": user_input})
                
                # print("-------Starting SQL Statement-------")
                # for row in response:
                #     print(row)
                # print("-------Ending SQL Statement-------")


                ### Replacing with Handle response function ###
                # if not response.startswith("SELECT"):
                #     response =response.split('\n',1)[1].split('`')[0]

                # if response.startswith("SELECT"):
                #     print(response)
                #     db_results = self.execute_queries(response)
                # else:
                #     db_results=temp_response

                # if db_results:
                #     return db_results
                # return "None"
                ### Replaced with Handle response function ###
                db_results = self.handle_response(response)
                formatted_result = {"text":db_results}
                # formatted_string = ', '.join([str(item) for sublist in db_results for item in sublist])
                if db_results:
                    self._conversation_history.append({"role": "AI Assistant", "content": db_results})
                    # for message in self._conversation_history:
                    #     print("History: ",message)
                    print("History: ",self._conversation_history[-2])
                    return f'{formatted_result}\n' + '---' * 20
                

            except Exception as e:
                print(f"Error executing database query: {e}")
                return f"An error occurred while processing your request. Please try again later."


        try:
            response = self.conversation.send_message(
                content=user_input,
                generation_config=self._generation_config(temperature)
            )
            response.resolve()
            self._conversation_history.append({"role": "AI Assistant", "content": response})
            return f'{response.text}\n' + '---' * 20
        except Exception as e:
            raise Exception(e.message)

    # Formatting Database result   
    def format_as_instructions(self, data):
        if data:
            formatted_data = [
                {
                    "Product Number": row[0],
                    "Product Name": row[1],
                    "Product Description": row[2],
                    "Product Back Office Code": row[12],
                    "Vision Number": row[15],
                    "Product UDF7": row[22],
                    "Product UDF8": row[23],
                    "Product Creation Date": row[29].strftime("%Y-%m-%d %H:%M:%S"),
                    "Product Last Update Date": row[30].strftime("%Y-%m-%d %H:%M:%S"),
                    "Product Has Item": row[31],
                    "Product ID": row[40],
                    "Product Band ID": row[42],
                    "UOMID": row[48],
                    "Product Created By User ID": row[49],
                    "Product Updated By User ID": row[50],
                    "Product Property 1ID": row[51],
                    "Component UMOID": row[58],
                    "Prodls Primary": row[62],
                    "Product HSN Code": row[69]
                }
                for row in data
            ]
            json_data = json.dumps(formatted_data, indent=4)
            return f"Here are the related products:\n{json_data}"
        return "No related products found."


    def execute_queries(self, prompt):
        with self.db_engine.connect() as conn:
            trans = conn.begin()
            try:
                result = conn.execute(
                    text(prompt),
                    # {"prodname": f"%{prompt}%"}
                ).fetchall()

                if result:
                    # for row in result:
                    #     print(row)
                    return result
                else:
                    return None

                trans.commit()
            except Exception as e:
                trans.rollback()
                raise GenAIExeption(f"Database Error: {str(e)}")

    
    @property
    def history(self,):
        conversation_history = [
            {'role' : message.role, 'text':message.parts[0].text} for message in self.conversation.history
        ]
        return conversation_history

    
    def clear_conversation(self):
        self.conversation = self.model.start_chat(history = [])
    
    def start_conversation(self):
        self.conversation = self.model.start_chat(history = self._conversation_history)
    
    def _construct_message(self,text,role='user'):
        return {'role' : role,'parts':[text]}
    
    def preload_conversation(self,conversation_history = None):
        if isinstance(conversation_history,list):
            self._conversation_history=conversation_history
        else:
            self._conversation_history = [
                self._construct_message('From now on, return the output as JSON object that can be loaded in Python with the key as \'text\'. For Example, {"text": "<output goes here>"}'),
                self._construct_message('{"text":"Sure, I can return the output as a regular JSON object with the key as `text`. Here is an example {"text":"Your Output"}.','model')
            ]


# # Getting Response as SQL Quary
# def get_gemini_response(question,prompt):
#     model=genai.GenerativeModel('gemini-1.5-flash')
#     response=model.generate_content([prompt[0],question])
#     return response.text

prompt=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name PRODUCTS and has the following columns - ProdNum, ProdName, ProdDesc , 
    ProdDateCreated etc. \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM PRODUCTS ;
    \nExample 2 - Tell me all the sky tone products?, 
    the SQL command will be something like this SELECT * FROM PRODUCTS 
    where ProdName LIKE '%sky tone%'; 
    also the sql code should not have ``` in beginning or end and sql word in output

    """
]