/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;

/**
 *
 * @author ameya
 */
import javax.swing.*;
import java.awt.*;
import java.sql.*;
import java.awt.event.*;
// import the fields class for forms making, sql for database
// awt and swing both for forms
// j panel is like div tag in html

// Creating a login form

public class Login_user extends JFrame implements ActionListener
{
    // declaring private variables
    private JPanel panel;
    // like div in html
    private JButton b1,b2,b3;
    private JPasswordField passwordField;
    private JTextField textField;
    private JLabel l2;
    // creating the form in a constructor
    public Login_user()
    {
        setLayout(null);
        // we are creating aa new layout not built in from swing
        setBackground(new Color(169,169,169));
        setBounds(600,300,600, 400);
        
        panel = new JPanel();
        // new Jpanel is the object stored in the variable panel
        panel.setBackground(Color.WHITE);
        setContentPane(panel);
        // adding objjects to the content pane container
        // this is the outer box
        panel.setLayout(null);
        
        
        // label and field
        
        JLabel l1 = new JLabel("UserName : ");
        l1.setBounds(124,89,95,24);
        panel.add(l1);
        // add objects to panel
        
        textField = new JTextField();
        textField.setBounds(210,93,157,20);
        panel.add(textField);
        
        l2 = new JLabel("Password: ");
        l2.setBounds(124, 124, 95, 24);
        // keep x and y cordinates x same if one below another and y same for one besides another
        panel.add(l2);
        
        passwordField = new JPasswordField();
        passwordField.setBounds(210,128, 157,20);
        panel.add(passwordField);    
        
        b1 = new JButton("Login");
        b1.addActionListener(this);
        b1.setBackground(new Color(26,139,187));//white kind of  color
        b1.setForeground(Color.red);
        b1.setBounds(149, 181, 113, 39);
        panel.add(b1);
        
        b2 = new JButton("Sign up");
        b2.addActionListener(this);
        b2.setBackground(new Color(26,139,187));//white kind of  color
        b2.setForeground(Color.red);
        b2.setBounds(289,181, 113, 39);
        panel.add(b2);
        
        b3 = new JButton("Forgot Password");
        b3.addActionListener(this);
        b3.setBackground(new Color(253,245, 230));//white kind of  color
        b3.setForeground(Color.red);
        b3.setBounds(199, 231, 179, 39);
        panel.add(b3);
        
        JLabel j5 = new JLabel("Trouble in login?");
        j5.setFont(new Font("Tahoma", Font.PLAIN, 15));
        j5.setBackground(new Color(26,139,87) );
        j5.setForeground(Color.red);
        j5.setBounds(70, 240, 130, 20);
        panel.add(j5); 
    }
    
    public void actionPerformed(ActionEvent ae){
        // checking which button is clicked
        // login button clicked
        if (ae.getSource() == b1)
        {
            Boolean status = false;
            // when select in sql then use execute statement
            // when insert in sql select in sql use add statement
            try{
                conn con_obj = new conn();
                String sql = "select * from account where username =? and password = ?";
                // replace ? by value entered by user in text field
                // using getText anf set String method
                PreparedStatement st =  con_obj.c.prepareStatement(sql);
                // All sql data in Prepared Statement 
                st.setString(1,textField.getText());
                st.setString(2,passwordField.getText());
                // now executing the query and storing the result in rs by each line
                ResultSet rs = st.executeQuery(sql);
                if (rs.next()){
                // if the username and password in each row then
                this.setVisible(false);
                new loading().setVisible(true);
                // my loading class object called
                }
                else{
                    JOptionPane.showMessageDialog(null, "invalid login");
                }
            }
            catch(Exception e2){
                e2.printStackTrace();
            }
        }
        //sign up page clicked
        if (ae.getSource() == b2){
            this.setVisible(false);
            // when sign up go to different page by stopping this current page
            new Signup().setVisible(true); // new class object sign up which is now shown
            
        }
        if (ae.getSource() == b3){
            this.setVisible(false);
           // new class Forgot
            new Forgot().setVisible(true);
        }
    
    }
    
    public static void main(String[] args)
    {
        new Login_user().setVisible(true);
    }
    
}
