import tree_sitter_languages
import random
import sys
import string
from copy import deepcopy
from datasets import load_dataset, Dataset, concatenate_datasets


# from transformers import AutoTokenizer, BertModel, AutoModelForSequenceClassification
# from peft import get_peft_config, get_peft_model, LoraConfig, Taskdata_type


class DeadCodeAugmentor:
    def __init__(self):
        # Initialize with available augmentation methods
        self.used_variable = []
        self.parser = tree_sitter_languages.get_parser("cpp")

    @staticmethod
    def get_number_data_types():
        return ["int", "long int ", "double", "float"]

    @staticmethod
    def get_random_string(n):
        return ''.join(random.choices(string.ascii_letters, k=n))

    def get_random_int(self, start, end):
        temp = random.randint(start, end)
        while temp in self.used_variable:
            temp = random.randint(start, end)
        return temp

    def number_assignment_augmentation(self, code):
        options = []
        for data_type in self.get_number_data_types():
            options.append(
                f"{data_type} variable_{self.get_random_int(0, sys.maxsize)} = {random.randint(0, sys.maxsize)};")
        return self.insert_code(code, random.choice(options))

    def string_assignment_augmentation(self, code):
        options = [
            f"std::string variable_{self.get_random_int(0, sys.maxsize)} = \"{self.get_random_string(random.randint(0, 10))}\";",
            f"std::string variable_{self.get_random_int(0, sys.maxsize)} (\"{self.get_random_string(random.randint(0, 10))}\");"]
        return self.insert_code(code, random.choice(options))

    def addition_augmentation(self, code):
        options = []
        for data_type in self.get_number_data_types():
            options.append(
                f"{data_type} variable_{self.get_random_int(0, sys.maxsize)} = {random.randint(1, sys.maxsize)} + ({random.randint(-sys.maxsize - 1, sys.maxsize)});")
            temp = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(-sys.maxsize - 1, sys.maxsize)};
            {data_type} variable_{self.get_random_int(0, sys.maxsize)} = variable_{temp} + {random.randint(1, sys.maxsize)};""")
            temp = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(-sys.maxsize - 1, sys.maxsize)};
            {data_type} variable_{self.get_random_int(0, sys.maxsize)} = {random.randint(1, sys.maxsize)} + variable_{temp};""")
        return self.insert_code(code, random.choice(options))

    def subtraction_augmentation(self, code):
        options = []
        for data_type in self.get_number_data_types():
            options.append(
                f"{data_type} variable_{self.get_random_int(0, sys.maxsize)} = {random.randint(-sys.maxsize - 1, -1)} - ({random.randint(-sys.maxsize - 1, sys.maxsize)});")
            temp = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(-sys.maxsize - 1, sys.maxsize)};
            {data_type} variable_{self.get_random_int(0, sys.maxsize)} = variable_{temp} - {random.randint(1, sys.maxsize - 1)};""")
            temp = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(-sys.maxsize - 1, sys.maxsize)};
            {data_type} variable_{self.get_random_int(0, sys.maxsize)} = -{random.randint(sys.maxsize - 1, sys.maxsize)} - variable_{temp};""")
        return self.insert_code(code, random.choice(options))

    def multiplication_augmentation(self, code):
        options = []
        for data_type in self.get_number_data_types():
            options.append(
                f"{data_type} variable_{self.get_random_int(0, sys.maxsize)} = {random.randint(-sys.maxsize - 1, sys.maxsize)} * ({random.randint(-sys.maxsize - 1, sys.maxsize)});")
            temp = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(-sys.maxsize - 1, sys.maxsize)};
            {data_type} variable_{self.get_random_int(0, sys.maxsize)} = variable_{temp} * ({random.randint(-sys.maxsize - 1, sys.maxsize)});""")
            temp = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(-sys.maxsize - 1, sys.maxsize)};
            {data_type} variable_{self.get_random_int(0, sys.maxsize)} = ({random.randint(-sys.maxsize - 1, sys.maxsize)}) * variable_{temp};""")
        return self.insert_code(code, random.choice(options))

    def insert_code(self, code, augmentation, line_no=None):
        code_lines = code.split("\n")
        if line_no is None:
            line_no = random.randint(1, len(code_lines) - 1)
        code_lines.insert(line_no, augmentation)
        return "".join(code_lines)

    def divide_augmentation(self, code):
        options = []
        for data_type in self.get_number_data_types():
            options.append(
                f"{data_type} variable_{self.get_random_int(0, sys.maxsize)} = {random.randint(-sys.maxsize - 1, sys.maxsize)} / ({random.randint(1, sys.maxsize)});")
            temp = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(-sys.maxsize - 1, sys.maxsize)};
            {data_type} variable_{self.get_random_int(0, sys.maxsize)} = variable_{temp} / ({random.randint(1, sys.maxsize)});""")
            temp = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(1, sys.maxsize)};
            {data_type} variable_{self.get_random_int(0, sys.maxsize)} = ({random.randint(-sys.maxsize - 1, sys.maxsize)}) * variable_{temp};""")
        return self.insert_code(code, random.choice(options))

    def additive_identity_augmentation(self, code):
        variable_dict = self.parse_variable_usage(code)
        if len(variable_dict) <= 0:
            return None
        else:
            line_no = random.choice(list(variable_dict.keys()))
            variable = random.choice(variable_dict[line_no])
            augmentation = f"{variable} = {variable} + 0;\n"
            return self.insert_code(code, augmentation, line_no=line_no + 1)

    def parse_variable_usage(self, code):
        line_no_dict = dict()

        def get_variables(node):
            if node.type == 'identifier' and node.parent and node.parent.type in ['parameter_declaration',
                                                                                  '_declarator', 'gnu_asm_expression',
                                                                                  'init_declarator', 'operator_cast',
                                                                                  'declaration']:
                if node.start_point[0] in line_no_dict:
                    line_no_dict[node.start_point[0]].append(code[node.start_byte: node.end_byte])
                else:
                    line_no_dict[node.start_point[0]] = [code[node.start_byte: node.end_byte]]
            for child in node.children:
                get_variables(child)

        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        get_variables(root_node)
        return line_no_dict

    def multiplicative_identity_augmentation(self, code):
        variable_dict = self.parse_variable_usage(code)
        if len(variable_dict) <= 0:
            return None
        else:
            line_no = random.choice(list(variable_dict.keys()))
            variable = random.choice(variable_dict[line_no])
            augmentation = f"{variable} = {variable} * 1;\n"
            return self.insert_code(code, augmentation, line_no=line_no + 1)

    def conditional_augmentation(self, code):
        options = []
        for data_type in self.get_number_data_types():
            temp = self.get_random_int(0, sys.maxsize)
            options.append(
                f"{data_type} variable_{temp} =  {random.randint(1, sys.maxsize)};\n  if (variable_{temp} < 0) {{\n variable_{temp} = {random.randint(-sys.maxsize - 1, sys.maxsize)};}}")
        return self.insert_code(code, random.choice(options))

    def loop_augmentation(self, code):
        options = []
        for data_type in self.get_number_data_types():
            temp = self.get_random_int(0, sys.maxsize)
            temp1 = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp1} = {random.randint(1, sys.maxsize)};
                                for ({data_type} variable_{temp} =  {random.randint(-sys.maxsize - 1, -1)}; variable_{temp} > 0; variable_{temp}--) {{
                                    variable_{temp1}--;
                                }}""")
            temp = self.get_random_int(0, sys.maxsize)
            temp1 = self.get_random_int(0, sys.maxsize)
            options.append(f"""{data_type} variable_{temp} = {random.randint(1, sys.maxsize)};
                               {data_type} variable_{temp1} = {random.randint(1, sys.maxsize)};
                                while (variable_{temp} > 0) {{
                                    variable_{temp1}--;
                            }}""")
        return self.insert_code(code, random.choice(options))

    def remove_comments(self, c_code):
        result = ''
        in_string = False
        in_single_line_comment = False
        in_multi_line_comment = False
        i = 0

        while i < len(c_code):
            # Check for start of string literal
            if c_code[i] == '"' and not in_single_line_comment and not in_multi_line_comment:
                in_string = not in_string
                result += c_code[i]
            # Check for single-line comment
            elif c_code[i:i + 2] == '//' and not in_string and not in_multi_line_comment:
                in_single_line_comment = True
            # Check for end of single-line comment
            elif c_code[i] == '\n' and in_single_line_comment:
                in_single_line_comment = False
                result += c_code[i]
            # Check for start of multi-line comment
            elif c_code[i:i + 2] == '/*' and not in_string and not in_single_line_comment:
                in_multi_line_comment = True
                i += 1  # Skip next char as it is part of the comment
            # Check for end of multi-line comment
            elif c_code[i:i + 2] == '*/' and in_multi_line_comment:
                in_multi_line_comment = False
                i += 1  # Skip next char as it is part of the comment
            # Add character to result if not in a comment
            elif not in_single_line_comment and not in_multi_line_comment:
                result += c_code[i]
            i += 1
        return result

    def extract_method_names(self, c_code):
        tree = self.parser.parse(bytes(c_code, "utf8"))
        root_node = tree.root_node

        method_names = []

        def process_node(node):
            if node.type == 'function_definition' or node.type == 'method_definition':
                for child in node.children:
                    # Look for the function declarator node
                    if child.type == 'function_declarator':
                        for identifier in child.children:
                            # The function name is typically an identifier
                            if identifier.type == 'identifier':
                                method_name = c_code[identifier.start_byte:identifier.end_byte]
                                method_names.append(method_name)
                                break

            for child in node.children:
                process_node(child)

        process_node(root_node)
        return method_names

    def not_member_random(self, existing_id):
        current_random_id = random.randint(1, sys.maxsize)
        while current_random_id in existing_id:
            current_random_id = random.randint(1, sys.maxsize)
        return current_random_id

    def normalize_code(self, c_code, random=False):
        tree = self.parser.parse(bytes(c_code, "utf8"))
        method_list = self.extract_method_names(c_code)
        root_node = tree.root_node
        if random:
            function_counter = self.get_random_int(1, sys.maxsize)
            variable_counter = self.get_random_int(1, sys.maxsize)
        else:
            function_counter = 1
            variable_counter = 1
        identifiers = {}
        new_code = list(c_code)

        def normalize_node(node, new_name):
            for i in range(node.start_byte, node.end_byte):
                new_code[i] = ""
            new_code[node.start_byte] = new_name

        def process_node(node):
            nonlocal function_counter, variable_counter
            if node.type == 'identifier':
                identifier = c_code[node.start_byte:node.end_byte]

                # Check if it's a function or a variable
                parent = node.parent
                if parent and parent.type in ['function_declarator', 'call_expression']:
                    if identifier in method_list:
                        # Function processing
                        if identifier not in identifiers:
                            identifiers[identifier] = f'function_{function_counter}'
                            if random:
                                function_counter = self.get_random_int(1, sys.maxsize)
                            else:
                                function_counter += 1
                        normalize_node(node, identifiers[identifier])
                else:
                    # Variable processing
                    if parent.type != 'qualified_identifier':
                        if identifier not in identifiers:
                            identifiers[identifier] = f'variable_{variable_counter}'
                            if random:
                                variable_counter = self.get_random_int(1, sys.maxsize)
                            else:
                                variable_counter += 1
                        normalize_node(node, identifiers[identifier])

            # Recursively process all children nodes
            for child in node.children:
                process_node(child)

        process_node(root_node)
        return ''.join(new_code)


# Example usag
if __name__ == "__main__":
    code = """void exampleMethod(int x, float y) {
        int a = 5;
        a = a + 1;
        int b = a * 2;
    }"""
    augmenter = DeadCodeAugmentor()
    print(augmenter.normalize_code(code))
