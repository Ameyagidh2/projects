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
import java.util.*;

public class ReadMe extends JFrame{
    private JLabel l1,l2,l3,l4;
    private  JTextArea t1;
    private JPanel controlPanel;
    
    public static void main(String[] args) {
        new ReadMe().setVisible(true);
    }
    
    public ReadMe(){
        super("Readme - Library Management System");
        setLayout(null);
        setBounds(30,300,900,300);
        setBackground(Color.WHITE);   
        
    controlPanel = new JPanel();
    controlPanel.setBackground(Color.WHITE);
    controlPanel.setBounds(30,100,300,300);
    setContentPane(controlPanel);
    
    l1 = new JLabel("Library");
    l1.setBounds(200,200,80,30);
    l1.setFont(new Font("Trebuchet Ms",Font.BOLD,35));
    l1.setForeground(Color.GREEN);
    controlPanel.add(l1);
    
    l2 = new JLabel("Management System\n");
    l2.setBounds(200,400,80,30);
    l2.setFont(new Font("Trebuchet Ms",Font.BOLD,25));
    l2.setForeground(Color.RED);
    controlPanel.add(l2);
    
    l3 = new JLabel("V.1.0\n");
    l3.setBounds(200,600,80,30);
    l3.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
    l3.setForeground(Color.BLUE);
    controlPanel.add(l3);
    
    l4 = new JLabel("Developed by Ameya\n");
    l4.setBounds(200,800,80,30);
    l4.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
    l4.setForeground(Color.BLUE);
    controlPanel.add(l4);
    
    t1 = new JTextArea("Library Mangement system\nHelps to organize the data of books and the students who issued and returned the books\nIt also gives information about the book statistics");
    t1.setBounds(30,900,80,30);
    t1.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
    t1.setForeground(Color.BLUE);
    controlPanel.add(t1);
        
    }
    
}
