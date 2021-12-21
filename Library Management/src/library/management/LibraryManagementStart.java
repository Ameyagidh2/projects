/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;

/**
 *
 * @author ameya
 */
import java.awt.*;
import javax.swing.*;
import java.sql.*;
import java.awt.event.*;
import javax.swing.border.*;

public class LibraryManagementStart extends JFrame implements ActionListener
{
    private JLabel l1;
    private JButton b1;
    private JPanel controlPanel;
    
    public static void main(String[] args) {
        new LibraryManagementStart().setVisible(true);
    }
    
 public LibraryManagementStart(){
    
     setSize(1366,390);
     setLayout(null);
     setLocation(300,300);  
     b1 = new JButton("next");
    ImageIcon i1 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/first.jpg"));
    Image i2 = i1.getImage().getScaledInstance(1366,309,Image.SCALE_DEFAULT);
    ImageIcon i3 = new ImageIcon(i2);
    l1 = new JLabel(i3);
    l1.setBounds(0,0,1366,309);
    b1.setBounds(1050,250,200,60);
    b1.addActionListener(this);
    l1.add(b1);
    add(l1);
         
}
 public void actionPerformed(ActionEvent ae){
    this.setVisible(false);
     new Login_user().setVisible(true);
     
 }
}
